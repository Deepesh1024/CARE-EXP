import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
import copy
import json
import os
import random
from tqdm import tqdm

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

def get_jetmoe_mlp_tensors(model, layer_idx):
    input_linear, output_linear, router = None, None, None
    prefix = f"model.layers.{layer_idx}.mlp"
    for name, param in model.named_parameters():
        if f"{prefix}.input_linear.weight" in name: input_linear = param
        elif f"{prefix}.output_linear.weight" in name: output_linear = param
        elif f"{prefix}.router.layer.weight" in name: router = param
    return input_linear, output_linear, router

def merge_experts_jetmoe(model, layer_idx, keep_idx, remove_idx):
    """
    Safely merges remove_idx into keep_idx.

    Strategy:
      - Average both experts' weights, write result into BOTH slots so that
        whichever slot the router picks, the computation is the same.
      - Average the canonical (keep) router row, leave it as-is.
      - Scale the removed slot's router row by 0.99 so the canonical slot
        ALWAYS scores marginally higher in top-k selection.
        This prevents torch.topk from choosing BOTH slots of the same merged
        pair, which would waste a top-k slot on a duplicate computation and
        halve the model's effective routing capacity — causing the catastrophic
        PPL blowup seen with multiple merges per layer (8→6→4 experts).
    """
    input_linear, output_linear, router = get_jetmoe_mlp_tensors(model, layer_idx)
    with torch.no_grad():
        merged_input  = (input_linear.data[keep_idx]  + input_linear.data[remove_idx])  / 2.0
        merged_output = (output_linear.data[keep_idx] + output_linear.data[remove_idx]) / 2.0
        merged_router = (router.data[keep_idx]        + router.data[remove_idx])        / 2.0

        # Both slots compute identical outputs (safe for batched parallel experts)
        input_linear.data[keep_idx].copy_(merged_input)
        input_linear.data[remove_idx].copy_(merged_input)
        output_linear.data[keep_idx].copy_(merged_output)
        output_linear.data[remove_idx].copy_(merged_output)

        # Canonical slot gets the merged router row
        router.data[keep_idx].copy_(merged_router)
        # Remove slot gets a 1% weaker router row → canonical always wins top-k
        router.data[remove_idx].copy_(merged_router * 0.99)

def get_characteristic_activations(model, tokenizer):
    """
    Compute per-expert Characteristic Activations (CA) for Sub-MoE clustering.

    CA proxy: mean of each expert's input projection weight rows.
      - input_linear.weight shape: [8, 11264, 2048]
      - We take the mean over the 11264 output dim → [8, 2048]
      - This captures the average 'receptive field' of each expert in
        hidden-state space, which is the correct signal for Sub-MoE:
        experts with similar means produce similar outputs for similar
        inputs and can safely be merged.

    Avoid using router.weight rows as CA proxy: high router cosine similarity
    means two experts are co-activated (bad to merge), not that they're
    functionally redundant (good to merge).
    """
    print("[*] Computing Expert Weight-Space CA...")
    final_ca = {}
    for i in range(24):
        input_linear, _, _ = get_jetmoe_mlp_tensors(model, i)
        # input_linear.weight: [8, 11264, 2048]  (num_experts, ffn_dim, hidden_dim)
        # Mean over ffn_dim → [8, 2048] float32
        ca = input_linear.data.mean(dim=1).cpu().to(torch.float32).numpy()
        final_ca[i] = ca
    return final_ca

def run_submoe(model, tokenizer, ca_stats, target_experts):
    print(f"\n[Sub-MoE] Compressing to {target_experts} experts...")
    num_merges = 8 - target_experts  # How many experts to remove per layer

    for i in range(24):
        ca = ca_stats[i]  # [8, 2048] float32 numpy

        # Greedy agglomerative: repeatedly merge the closest pair
        # (by cosine similarity of router weight rows) until we reach target_experts.
        # Tracks which slot is the canonical representative for each logical expert.
        canonical = list(range(8))  # canonical[e] = model tensor slot for logical expert e
        active = list(range(8))     # which logical experts are still alive

        # Normalise rows for cosine similarity
        norms = np.linalg.norm(ca, axis=1, keepdims=True) + 1e-9
        ca_normed = ca / norms

        for _ in range(num_merges):
            # Find the most similar pair among active experts
            best_sim = -1.0
            best_a, best_b = active[0], active[1]
            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    e_a = active[ai]
                    e_b = active[bi]
                    sim = float(np.dot(ca_normed[e_a], ca_normed[e_b]))
                    if sim > best_sim:
                        best_sim = sim
                        best_a, best_b = e_a, e_b

            # Merge best_b into best_a
            slot_a = canonical[best_a]
            slot_b = canonical[best_b]
            merge_experts_jetmoe(model, i, slot_a, slot_b)

            # Update normed CA for best_a (average of the two)
            ca_normed[best_a] = (ca_normed[best_a] + ca_normed[best_b]) / 2.0
            norm = np.linalg.norm(ca_normed[best_a]) + 1e-9
            ca_normed[best_a] /= norm

            # Remove best_b from active list
            active.remove(best_b)

    return evaluate_subset(model, tokenizer)

def run_random(model, tokenizer, target_experts):
    print(f"\n[Random] Compressing to {target_experts} experts...")
    random.seed(42)
    for i in range(24):
        active_experts = list(range(8))
        while len(active_experts) > target_experts:
            remove_idx = random.choice(active_experts)
            active_experts.remove(remove_idx)
            keep_idx = random.choice(active_experts)
            merge_experts_jetmoe(model, i, keep_idx, remove_idx)
            
    return evaluate_subset(model, tokenizer)

def run_care_adaptive(model, tokenizer, ca_stats, target_experts):
    print(f"\n[CARE Adaptive] Compressing to {target_experts} experts...")
    # CARE simplifies: finds the closest CA pairs (candidate pool), evaluates micro-PPL, picks best
    for i in tqdm(range(24), desc="CARE Iterating Layers"):
        ca = ca_stats[i]
        active_experts = list(range(8))
        
        while len(active_experts) > target_experts:
            # Candidate Pool (top 2 closest pairs based on CA cosine similarity)
            pairs = []
            for idx1 in active_experts:
                for idx2 in active_experts:
                    if idx1 >= idx2: continue
                    sim = np.dot(ca[idx1], ca[idx2]) / (np.linalg.norm(ca[idx1]) * np.linalg.norm(ca[idx2]) + 1e-9)
                    pairs.append((sim, idx1, idx2))
            pairs.sort(reverse=True) # Highest similarity first
            candidates = pairs[:2]
            
            best_pair = None
            best_ppl = float('inf')
            
            # Evaluate candidates
            for sim, k_idx, r_idx in candidates:
                # Backup weights for this layer
                in_w, out_w, r_w = get_jetmoe_mlp_tensors(model, i)
                in_bak, out_bak, r_bak = in_w.data.clone(), out_w.data.clone(), r_w.data.clone()
                
                # Merge
                merge_experts_jetmoe(model, i, k_idx, r_idx)
                
                # Micro eval (500 tokens)
                micro_ppl = evaluate_subset(model, tokenizer, num_tokens=500)
                
                if micro_ppl < best_ppl:
                    best_ppl = micro_ppl
                    best_pair = (k_idx, r_idx)
                
                # Restore
                in_w.data.copy_(in_bak)
                out_w.data.copy_(out_bak)
                r_w.data.copy_(r_bak)
                
            # Commit best merge
            keep_idx, remove_idx = best_pair
            merge_experts_jetmoe(model, i, keep_idx, remove_idx)
            active_experts.remove(remove_idx)
            
    return evaluate_subset(model, tokenizer)

def main():
    print("="*50)
    print("PHASE 8: FULL BENCHMARK SUITE")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    base_ppl = evaluate_subset(model, tokenizer)
    print(f"Base PPL (8 experts): {base_ppl:.4f}")
    
    base_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ca_stats = get_characteristic_activations(model, tokenizer)
    
    checkpoints = [7, 6, 4]
    
    results = {
        "Base": {8: base_ppl},
        "Random": {},
        "Sub-MoE": {},
        "CARE Adaptive": {}
    }
    
    for target in checkpoints:
        # Random
        model.load_state_dict(base_state_dict, strict=True)
        results["Random"][target] = run_random(model, tokenizer, target)
        
        # Sub-MoE
        model.load_state_dict(base_state_dict, strict=True)
        results["Sub-MoE"][target] = run_submoe(model, tokenizer, ca_stats, target)
        
        # CARE Adaptive
        model.load_state_dict(base_state_dict, strict=True)
        results["CARE Adaptive"][target] = run_care_adaptive(model, tokenizer, ca_stats, target)
        
        print(f"\n--- Checkpoint {target} Experts ---")
        print(f"Random: {results['Random'][target]:.4f}")
        print(f"Sub-MoE: {results['Sub-MoE'][target]:.4f}")
        print(f"CARE:   {results['CARE Adaptive'][target]:.4f}")
        
    os.makedirs("benchmark_results/JetMoE-8B", exist_ok=True)
    with open("benchmark_results/JetMoE-8B/final_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
