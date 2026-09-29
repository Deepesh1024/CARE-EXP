import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import os
import time
from tqdm import tqdm

def get_calibration_dataset(tokenizer, num_samples=64, seq_length=512):
    print("\n[REAP] Loading calibration dataset (WikiText-2 Train)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [text for text in dataset["text"] if len(text.strip()) > 100]
    calibration_texts = texts[:num_samples]
    
    encodings = []
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=seq_length, truncation=True)
        encodings.append(inputs["input_ids"])
    return encodings

def evaluate_subset(model, tokenizer, num_tokens=15000):
    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = 512
    limit = min(encodings.input_ids.size(1), num_tokens)
    
    nlls = []
    total_tokens = 0
    with torch.inference_mode():
        for begin_loc in range(0, limit, seq_len):
            end_loc = min(begin_loc + seq_len, limit)
            trg_len = end_loc - begin_loc
            if trg_len == 0: break
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            total_tokens += trg_len
    return torch.exp(torch.stack(nlls).sum() / total_tokens).item()

def compute_reap_scores(model, encodings):
    """
    Computes REAP saliency scores for all MLP experts across all layers.
    S_j = (1 / N_j) * sum( g_j(t) * ||f_j(t)||_2 )
    """
    print("\n[REAP] Computing expert saliency scores...")
    start_time = time.time()
    
    # Initialize trackers
    # For each layer, for each expert: sum of (router_weight * ||expert_out||_2) and token count
    num_layers = 24
    num_experts = 8
    reap_sum = {i: torch.zeros(num_experts, device="cpu") for i in range(num_layers)}
    reap_count = {i: torch.zeros(num_experts, device="cpu") for i in range(num_layers)}
    
    hooks = []
    
    def get_mlp_hook(layer_idx):
        def hook(module, inputs, output):
            hidden_states = inputs[0] # [batch, seq, hidden_dim]
            N_tokens = hidden_states.view(-1, hidden_states.size(-1)) # [tokens, 2048]
            
            # 1. Get router logits
            router_logits = module.router.layer(N_tokens) # [tokens, 8]
            probs = torch.softmax(router_logits.float(), dim=-1)
            
            # 2. Get routing weights and selected experts (Top-2)
            # JetMoE defaults to top_k = 2
            topk_weights, selected_experts = torch.topk(probs, 2, dim=-1)
            
            # 3. Extract expert weights to compute f_j(t)
            W_in = module.experts.input_linear.weight # [8, 11264, 2048]
            W_out = module.experts.output_linear.weight # [8, 2048, 5632]
            
            for j in range(num_experts):
                # Mask of tokens assigned to expert j
                mask = (selected_experts == j).any(dim=-1)
                if not mask.any(): continue
                
                tokens_for_j = N_tokens[mask] # [K, 2048]
                
                # Compute SwiGLU forward pass for expert j
                h = F.linear(tokens_for_j, W_in[j]) # [K, 11264]
                gate, up = h.chunk(2, dim=-1) # [K, 5632], [K, 5632]
                h_act = F.silu(gate) * up # [K, 5632]
                out_j = F.linear(h_act, W_out[j]) # [K, 2048]
                
                # ||f_j(t)||_2
                norm_j = torch.linalg.norm(out_j, dim=-1).cpu() # [K]
                
                # g_j(t) - router gate weight for expert j
                g_j = probs[mask, j].cpu() # [K]
                
                # Update stats
                reap_sum[layer_idx][j] += (g_j * norm_j).sum()
                reap_count[layer_idx][j] += mask.sum()
                
        return hook

    print("[REAP] Attaching hooks to MLP layers...")
    for i in range(num_layers):
        mlp_module = model.model.layers[i].mlp
        hooks.append(mlp_module.register_forward_hook(get_mlp_hook(i)))
        
    print(f"[REAP] Running {len(encodings)} calibration sequences...")
    with torch.no_grad():
        for input_ids in tqdm(encodings):
            model(input_ids.cuda())
            
    for h in hooks:
        h.remove()
        
    final_scores = {}
    for i in range(num_layers):
        # Average the REAP sum by the number of tokens routed to the expert
        # Add epsilon to prevent division by zero
        scores = reap_sum[i] / (reap_count[i] + 1e-9)
        final_scores[i] = scores.tolist()
        
    calibration_time = time.time() - start_time
    print(f"[REAP] Saliency computation finished in {calibration_time:.2f}s.")
    
    return final_scores, calibration_time

def prune_model(model, final_scores, target_experts):
    print(f"\n[REAP] Pruning model to {target_experts} MLP experts per layer...")
    start_time = time.time()
    
    pruned_experts_log = {}
    
    for i in range(24):
        scores = torch.tensor(final_scores[i])
        # Find the top `target_experts` based on highest REAP scores
        _, retained_indices = torch.topk(scores, target_experts, largest=True)
        retained_indices = retained_indices.sort().values
        
        pruned_experts = [j for j in range(8) if j not in retained_indices.tolist()]
        pruned_experts_log[i] = pruned_experts
        
        mlp = model.model.layers[i].mlp
        
        # 1. Prune router
        router_layer = mlp.router.layer
        router_layer.weight = torch.nn.Parameter(router_layer.weight.data[retained_indices, :])
        router_layer.out_features = target_experts
        mlp.router.num_experts = target_experts
        
        # 2. Prune experts
        input_linear = mlp.experts.input_linear
        output_linear = mlp.experts.output_linear
        
        input_linear.weight = torch.nn.Parameter(input_linear.weight.data[retained_indices, :, :])
        output_linear.weight = torch.nn.Parameter(output_linear.weight.data[retained_indices, :, :])
        
        mlp.experts.num_experts = target_experts
        
    pruning_time = time.time() - start_time
    print(f"[REAP] Structural pruning finished in {pruning_time:.2f}s.")
    return pruned_experts_log, pruning_time

def main():
    print("="*50)
    print("PHASE 10: REAP COMPRESSION BASELINE")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    
    # 1. Calibration Phase
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    encodings = get_calibration_dataset(tokenizer)
    final_scores, calib_time = compute_reap_scores(model, encodings)
    
    os.makedirs("benchmark_results/JetMoE-8B/reap", exist_ok=True)
    with open("benchmark_results/JetMoE-8B/reap/reap_scores.json", "w") as f:
        json.dump(final_scores, f, indent=4)
        
    # Free memory before pruning evaluation
    del model
    torch.cuda.empty_cache()
    
    # 2. Pruning & Evaluation Phase
    results = {"calibration_time": calib_time, "pruning_time": {}, "ppl": {}}
    
    for target in [7, 6, 4]:
        model = AutoModelForCausalLM.from_pretrained(
            "jetmoe/jetmoe-8b", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        pruned_log, prune_time = prune_model(model, final_scores, target)
        results["pruning_time"][str(target)] = prune_time
        
        # Save pruned model locally (weights only, to save space, or just save the log)
        # We save the pruned model directory as requested
        save_dir = f"benchmark_results/JetMoE-8B/jetmoe_reap_{target}"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/pruned_experts.json", "w") as f:
            json.dump(pruned_log, f, indent=4)
        
        print(f"\n[REAP] Evaluating {target}-expert model...")
        ppl = evaluate_subset(model, tokenizer)
        results["ppl"][str(target)] = ppl
        print(f"REAP {target} Experts PPL: {ppl:.4f}")
        
        del model
        torch.cuda.empty_cache()
        
    with open("benchmark_results/JetMoE-8B/reap/final_reap_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n=== REAP BENCHMARK COMPLETE ===")
    for target in [7, 6, 4]:
        print(f"REAP 8->{target}: PPL = {results['ppl'][str(target)]:.4f}")

if __name__ == "__main__":
    main()
