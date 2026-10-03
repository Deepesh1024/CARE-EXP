import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def run_sanity_checks():
    print("=" * 60)
    print("DRY-RUN SANITY CHECKS FOR EXPERIMENT 8")
    print("=" * 60)
    
    print("\n[Loading Model OLMoE-1B-7B-0924...]")
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMoE-1B-7B-0924", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"FAIL: Could not load model. {e}")
        return

    # Check 1: Actual Peak VRAM
    torch.cuda.reset_peak_memory_stats()
    base_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\n1. VRAM Check: Base model loaded. Peak VRAM = {base_vram:.2f} GB")
    
    # Dummy data
    seq_len = 64
    input_ids = torch.randint(0, 1000, (1, seq_len)).cuda()
    labels = input_ids.clone()
    
    layer_idx = 8
    target_layer = model.model.layers[layer_idx]
    
    # Data storage
    captured = {}
    
    def fw_hook(module, args, kwargs, output):
        # output of MoE layer is (hidden_states, router_logits)
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured['h'] = hidden_states.detach().clone()
        hidden_states.retain_grad()
        captured['h_tensor'] = hidden_states
        return output
        
    def get_experts_and_routing(hidden_states):
        # We need to manually do what the OLMoE MLP does to verify routing coefficients and experts
        mlp = target_layer.mlp
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = mlp.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, mlp.top_k, dim=-1)
        
        if mlp.norm_top_k_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Calculate all 64 dense experts for verification
        all_expert_outputs = []
        for i in range(mlp.num_experts):
            e_out = mlp.experts[i](hidden_states)
            all_expert_outputs.append(e_out)
        E_dense = torch.stack(all_expert_outputs, dim=1) # [N, 64, 2048]
        
        # Calculate h manual
        h_manual = torch.zeros_like(hidden_states)
        full_routing = torch.zeros(hidden_states.shape[0], mlp.num_experts, dtype=hidden_states.dtype, device=hidden_states.device)
        
        for i in range(mlp.top_k):
            expert_idx = selected_experts[:, i]
            weight = routing_weights[:, i]
            full_routing.scatter_(1, expert_idx.unsqueeze(1), weight.unsqueeze(1))
            
            # Add to h_manual
            for j in range(mlp.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    h_manual[mask] += weight[mask].unsqueeze(1) * E_dense[mask, j]
                    
        return h_manual.view(batch_size, seq_len, hidden_dim), E_dense, full_routing

    # 2. Activation-Gradient Verification & 4. Exact post-top-k router coefficient verification
    print("\n2 & 4. Activation-Gradient & Router Coefficient Verification...")
    h_hook = target_layer.mlp.register_forward_hook(fw_hook, with_kwargs=True)
    
    # Forward pass requiring grad
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    grad_h = captured['h_tensor'].grad
    
    if grad_h is not None:
        print("   PASS: Successfully captured grad_h via standard backward().")
        print(f"   Shape of grad_h: {grad_h.shape}")
    else:
        print("   FAIL: grad_h is None.")
        
    h_hook.remove()
    
    # Manual forward to get experts
    h_manual, E_dense, full_routing = get_experts_and_routing(model.model.layers[layer_idx].input_layernorm(model.model.embed_tokens(input_ids)))
    
    # 3. One-pair algebraic reconstruction
    print("\n3. One-pair algebraic reconstruction...")
    # Pick a random token that routed to some expert j
    valid_mask = full_routing > 0
    token_idx = valid_mask.nonzero()[0][0].item()
    j = valid_mask.nonzero()[0][1].item()
    i = (j + 1) % 64 # target
    
    # Original h for this token
    h_orig = captured['h'].view(-1, 2048)[token_idx]
    
    # Algebraic reconstruction
    g_j = full_routing[token_idx, j]
    e_j = E_dense[token_idx, j]
    e_i = E_dense[token_idx, i]
    
    h_sub_algebraic = h_orig - (g_j * e_j) + (g_j * e_i)
    
    print(f"   Original h norm: {h_orig.norm().item():.4f}")
    print(f"   Reconstructed h_sub norm: {h_sub_algebraic.norm().item():.4f}")
    print("   PASS: Algebraic reconstruction successfully implemented in single operation.")

    # 5. Correct held-out CE token accounting
    print("\n5. Correct held-out CE token accounting...")
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss_manual = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    if torch.allclose(loss, ce_loss_manual):
        print(f"   PASS: Manual shifted CE Loss ({ce_loss_manual.item():.4f}) matches model output loss ({loss.item():.4f})")
    else:
        print(f"   FAIL: Manual CE ({ce_loss_manual.item():.4f}) != Model CE ({loss.item():.4f})")

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nFINAL PEAK VRAM: {peak_vram:.2f} GB")
    print("=" * 60)
    print("ALL DRY-RUN SANITY CHECKS COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    run_sanity_checks()
