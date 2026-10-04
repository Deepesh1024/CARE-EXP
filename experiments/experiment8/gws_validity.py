import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import os
from tqdm import tqdm
import gc

def main():
    print("Loading model and tokenizer...")
    model_name = "allenai/OLMoE-1B-7B-0924"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    seq_len = 1024
    num_sequences = encodings.input_ids.size(1) // seq_len
    all_input_ids = encodings.input_ids[0, :num_sequences * seq_len].view(num_sequences, seq_len)

    split_A = all_input_ids[:256]
    split_B = all_input_ids[256:512]

    # --- SPLIT A: Compute GWS ---
    print("\n--- SPLIT A: Computing GWS ---")
    S_all = []
    G_all = []

    target_layer = model.model.layers[8]

    def get_full_experts_and_routing(hidden_states, mlp_module):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = mlp_module.gate(hidden_states)
        if isinstance(router_logits, tuple):
            router_logits = router_logits[0]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, model.config.num_experts_per_tok, dim=-1)
        
        if getattr(model.config, 'norm_topk_prob', False):
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        all_expert_outputs = []
        for k in range(model.config.num_experts):
            weight_gate_up = mlp_module.experts.gate_up_proj[k]
            weight_down = mlp_module.experts.down_proj[k]
            
            gate_up = F.linear(hidden_states, weight_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = mlp_module.experts.act_fn(gate) * up
            e_out = F.linear(intermediate, weight_down)
            
            all_expert_outputs.append(e_out)
        E_dense = torch.stack(all_expert_outputs, dim=1)
        
        full_routing = torch.zeros(hidden_states.shape[0], model.config.num_experts, dtype=hidden_states.dtype, device=hidden_states.device)
        for k in range(model.config.num_experts_per_tok):
            expert_idx = selected_experts[:, k]
            weight = routing_weights[:, k]
            full_routing.scatter_(1, expert_idx.unsqueeze(1), weight.unsqueeze(1))
            
        return E_dense, full_routing

    print("Running Split A forward/backward passes...")
    for i in tqdm(range(len(split_A))):
        input_ids = split_A[i:i+1].cuda()
        labels = input_ids.clone()
        
        captured = {}
        
        def pre_fw_hook(module, args):
            captured['mlp_input'] = args[0].detach().clone()
            return args
            
        def fw_hook(module, args, kwargs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.detach().requires_grad_(True)
            captured['h_tensor'] = h
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
            
        h1 = target_layer.mlp.register_forward_pre_hook(pre_fw_hook)
        h2 = target_layer.mlp.register_forward_hook(fw_hook, with_kwargs=True)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        grad_h = captured['h_tensor'].grad
        
        E_dense, full_routing = get_full_experts_and_routing(captured['mlp_input'], target_layer.mlp)
        
        grad_h_flat = grad_h.view(-1, 2048).float()
        E_dense_flat = E_dense.float()
        
        S = torch.einsum('nd,ned->ne', grad_h_flat, E_dense_flat)
        
        S_all.append(S.cpu())
        G_all.append(full_routing.cpu())
        
        h1.remove()
        h2.remove()
        
        model.zero_grad(set_to_none=True)

    S_all = torch.cat(S_all, dim=0)
    G_all = torch.cat(G_all, dim=0)

    print("Computing GWS matrix, RW-L2, and Usage...")
    gws_matrix = torch.zeros(64, 64)
    rw_l2_matrix = torch.zeros(64, 64)
    usage_vector = torch.zeros(64)
    
    for j in range(64):
        g_j = G_all[:, j]
        valid = g_j > 0
        usage_vector[j] = g_j.mean().item()
        
        if valid.sum() == 0:
            continue
        
        g_j_valid = g_j[valid]
        S_j = S_all[valid, j]
        
        for i in range(64):
            if i == j:
                continue
            S_i = S_all[valid, i]
            damage = (g_j_valid * (S_i - S_j)).sum() / len(G_all)
            gws_matrix[j, i] = damage.item()
            
            # Since we don't have full E_dense stored, we can compute RW-L2 dynamically during Split B...
            # Wait, S is just a dot product. RW-L2 requires the L2 norm of (e_i - e_j).
            # We must compute RW-L2 during Split B when we have access to e_i and e_j.

    # --- SPLIT B: Compute Actual Damage ---
    print("\n--- SPLIT B: Computing Actual Damage ---")
    cached_layer7_out = []
    baseline_ce_list = []

    captured = {}
    def pre_layer8_hook(module, args):
        captured['layer7_out'] = args[0].detach().cpu()
        return args

    h3 = target_layer.register_forward_pre_hook(pre_layer8_hook)

    # First baseline run
    print("Caching layer 7 outputs and computing baseline CE_1...")
    baseline_ce_1_total = 0
    baseline_tokens = 0
    for i in tqdm(range(len(split_B))):
        input_ids = split_B[i:i+1].cuda()
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum').item()
            baseline_ce_1_total += ce
            baseline_tokens += shift_labels.numel()
            cached_layer7_out.append(captured['layer7_out'])

    h3.remove()
    baseline_ce_1 = baseline_ce_1_total / baseline_tokens

    # Second baseline run
    print("Computing baseline CE_2 for noise floor...")
    baseline_ce_2_total = 0
    for i in tqdm(range(len(split_B))):
        input_ids = split_B[i:i+1].cuda()
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum').item()
            baseline_ce_2_total += ce

    baseline_ce_2 = baseline_ce_2_total / baseline_tokens
    noise_floor = abs(baseline_ce_1 - baseline_ce_2)
    
    print(f"Baseline CE_1: {baseline_ce_1:.7f}")
    print(f"Baseline CE_2: {baseline_ce_2:.7f}")
    print(f"Noise floor (absolute difference): {noise_floor:.7f}")
    
    baseline_ce = baseline_ce_1

    print("Truncating model to layers 8-15...")
    model.model.layers = model.model.layers[8:]
    truncated_target_layer = model.model.layers[0]

    cached_batches = []
    batch_size = 32
    for b in range(0, len(split_B), batch_size):
        embeds = torch.cat(cached_layer7_out[b:b+batch_size], dim=0).cuda()
        lbls = split_B[b:b+batch_size].cuda()
        cached_batches.append((embeds, lbls))

    def get_specific_experts_and_routing(hidden_states, mlp_module, j, i):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = mlp_module.gate(hidden_states)
        if isinstance(router_logits, tuple):
            router_logits = router_logits[0]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, model.config.num_experts_per_tok, dim=-1)
        
        if getattr(model.config, 'norm_topk_prob', False):
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        full_routing = torch.zeros(hidden_states.shape[0], model.config.num_experts, dtype=hidden_states.dtype, device=hidden_states.device)
        for k in range(model.config.num_experts_per_tok):
            expert_idx = selected_experts[:, k]
            weight = routing_weights[:, k]
            full_routing.scatter_(1, expert_idx.unsqueeze(1), weight.unsqueeze(1))
            
        def get_e(k):
            weight_gate_up = mlp_module.experts.gate_up_proj[k]
            weight_down = mlp_module.experts.down_proj[k]
            gate_up = F.linear(hidden_states, weight_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = mlp_module.experts.act_fn(gate) * up
            return F.linear(intermediate, weight_down)
            
        e_j = get_e(j)
        e_i = get_e(i)
            
        return e_j, e_i, full_routing

    global current_j, current_i, hook_rw_l2_sum
    current_j = -1
    current_i = -1
    hook_rw_l2_sum = 0

    def sub_fw_hook(module, args, kwargs, output):
        global hook_rw_l2_sum
        hidden_states = output[0] if isinstance(output, tuple) else output
        mlp_input = args[0]
        
        e_j, e_i, full_routing = get_specific_experts_and_routing(mlp_input, module, current_j, current_i)
        
        g_j = full_routing[:, current_j].unsqueeze(1)
        mask = (g_j > 0).to(hidden_states.dtype)
        
        # Calculate RW-L2 for this batch
        with torch.no_grad():
            l2_dist = torch.sum((e_i - e_j)**2, dim=-1, keepdim=True)
            hook_rw_l2_sum += (mask * g_j * l2_dist).sum().item()
            
        delta = mask * g_j * (e_i - e_j)
        
        h_new = hidden_states + delta.view(*hidden_states.shape)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    sub_hook = truncated_target_layer.mlp.register_forward_hook(sub_fw_hook, with_kwargs=True)

    pairs = [(j, i) for j in range(64) for i in range(64) if i != j]
    results = []

    print("Evaluating 4032 substitutions...")
    for j, i in tqdm(pairs):
        current_j = j
        current_i = i
        
        total_ce = 0
        total_tokens = 0
        total_rw_l2 = 0
        
        for embeds, lbls in cached_batches:
            with torch.no_grad():
                # Avoid passing labels to prevent HF from allocating massive float32 loss tensors internally
                outputs = model(inputs_embeds=embeds)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = lbls[..., 1:].contiguous()
                
                ce = F.cross_entropy(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum').item()
                total_ce += ce
                total_tokens += shift_labels.numel()
                
                # Retrieve the rw_l2 tracking from the hook using a global variable
                global hook_rw_l2_sum
                total_rw_l2 += hook_rw_l2_sum
                hook_rw_l2_sum = 0
                
                del outputs, shift_logits
                torch.cuda.empty_cache()
                
        avg_ce = total_ce / total_tokens
        damage = avg_ce - baseline_ce
        avg_rw_l2 = total_rw_l2 / total_tokens
        
        results.append({
            'j': j,
            'i': i,
            'gws': gws_matrix[j, i].item(),
            'gws_squared': (gws_matrix[j, i].item()) ** 2,
            'rw_l2': avg_rw_l2,
            'usage': usage_vector[j].item(),
            'actual_damage': damage
        })

        # Checkpoint every 50 pairs to prevent ANY data loss in the future
        if len(results) % 50 == 0:
            with open("experiments/experiment8/results_partial.json", "w") as f:
                json.dump(results, f, indent=4)

    with open("experiments/experiment8/results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment 8 Complete! Results saved to experiments/experiment8/results.json")
    
    # Calculate reporting stats
    actual_damages = [r['actual_damage'] for r in results]
    unique_vals = len(set(actual_damages))
    
    print(f"\nUnique actual_damage values: {unique_vals}")
    print(f"Min: {min(actual_damages):.7f}")
    print(f"Max: {max(actual_damages):.7f}")
    print(f"Median: {np.median(actual_damages):.7f}")
    print(f"Std: {np.std(actual_damages):.7f}")
    print(f"Noise floor: {noise_floor:.7f}")
    
    above_noise = sum(1 for d in actual_damages if abs(d) > noise_floor)
    print(f"Pairs above noise floor: {above_noise} / {len(actual_damages)}")
    
if __name__ == "__main__":
    main()
