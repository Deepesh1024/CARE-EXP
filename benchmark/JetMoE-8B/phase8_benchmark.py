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
    """Physically merges remove_idx into keep_idx and pseudo-removes remove_idx."""
    input_linear, output_linear, router = get_jetmoe_mlp_tensors(model, layer_idx)
    with torch.no_grad():
        merged_input = (input_linear.data[keep_idx] + input_linear.data[remove_idx]) / 2.0
        merged_output = (output_linear.data[keep_idx] + output_linear.data[remove_idx]) / 2.0
        merged_router = (router.data[keep_idx] + router.data[remove_idx]) / 2.0
        
        input_linear.data[keep_idx].copy_(merged_input)
        output_linear.data[keep_idx].copy_(merged_output)
        router.data[keep_idx].copy_(merged_router)
        
        input_linear.data[remove_idx].zero_()
        output_linear.data[remove_idx].zero_()
        router.data[remove_idx].fill_(-1e4) # Mask out

def get_characteristic_activations(model, tokenizer):
    print("[*] Collecting Characteristic Activations (CA)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [text for text in dataset["text"] if len(text.strip()) > 100][:32]
    
    ca_stats = {i: [] for i in range(24)}
    hooks = []
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            hidden = input[0].detach()
            ca_stats[layer_idx].append(hidden.view(-1, hidden.shape[-1]).cpu())
        return hook

    for i in range(24):
        router_module = model.model.layers[i].mlp.router
        hooks.append(router_module.register_forward_hook(get_hook(i)))
        
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model(**inputs)
            
    for h in hooks: h.remove()
    
    final_ca = {}
    for i in range(24):
        _, _, router = get_jetmoe_mlp_tensors(model, i)
        final_ca[i] = router.data.cpu().to(torch.float32).numpy() # Proxy CA
    return final_ca

def run_submoe(model, tokenizer, ca_stats, target_experts):
    print(f"\n[Sub-MoE] Compressing to {target_experts} experts...")
    for i in range(24):
        ca = ca_stats[i]
        kmeans = KMeans(n_clusters=target_experts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(ca)
        
        merged_clusters = {}
        for exp_idx, cluster_idx in enumerate(labels):
            if cluster_idx not in merged_clusters:
                merged_clusters[cluster_idx] = exp_idx
            else:
                keep_idx = merged_clusters[cluster_idx]
                merge_experts_jetmoe(model, i, keep_idx, exp_idx)
                
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
