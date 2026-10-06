import torch
import numpy as np
import json
import os
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from experiments.experiment4.data_loader import load_all

torch.set_grad_enabled(False)

def get_subspace_overlap(C_A, C_B, k_A, k_B):
    # C_A, C_B are covariance matrices. Find top eigenvectors
    eigvals_A, eigvecs_A = torch.linalg.eigh(C_A)
    eigvals_B, eigvecs_B = torch.linalg.eigh(C_B)
    
    # eigh returns ascending order. Get top k
    V_A = eigvecs_A[:, -k_A:] if k_A > 0 else torch.zeros((C_A.shape[0], 0), device=C_A.device)
    V_B = eigvecs_B[:, -k_B:] if k_B > 0 else torch.zeros((C_B.shape[0], 0), device=C_B.device)
    
    if k_A == 0 or k_B == 0:
        return 0.0
        
    # Cosines of principal angles are singular values of V_A^T V_B
    M = V_A.T @ V_B
    S = torch.linalg.svdvals(M)
    # Mean subspace overlap
    overlap = (S ** 2).sum().item() / max(k_A, k_B)
    return overlap

def get_k_for_fidelity(eigvals, total_variance, epsilon):
    # eigvals are sorted ascending
    target_error = epsilon * total_variance
    
    cumulative_sum = torch.cumsum(eigvals, dim=0)
    valid_indices = torch.where(cumulative_sum <= target_error)[0]
    if len(valid_indices) == 0:
        return len(eigvals)
    
    discard_count = valid_indices[-1].item() + 1
    k = len(eigvals) - discard_count
    return max(0, k)

def compute_frc_all_eps(C_A, C_B, epsilons):
    # Independent costs
    eigvals_A, _ = torch.linalg.eigh(C_A)
    eigvals_B, _ = torch.linalg.eigh(C_B)
    
    eigvals_A = torch.clamp(eigvals_A, min=0.0)
    eigvals_B = torch.clamp(eigvals_B, min=0.0)
    
    var_A = eigvals_A.sum().item()
    var_B = eigvals_B.sum().item()
    
    res = {}
    for eps in epsilons:
        if var_A == 0 or var_B == 0:
            res[str(eps)] = {"k_A": 0, "k_B": 0, "optimal_cost": 0, "k_shared": 0, "k_res_A": 0, "k_res_B": 0, "overlap": 0.0}
            continue
            
        k_A = get_k_for_fidelity(eigvals_A, var_A, eps)
        k_B = get_k_for_fidelity(eigvals_B, var_B, eps)
        res[str(eps)] = {"k_A": k_A, "k_B": k_B, "optimal_cost": float('inf'), "k_shared": 0, "k_res_A": 0, "k_res_B": 0}
        
    if var_A == 0 or var_B == 0:
        return res
        
    # Joint SVD
    C_AB = C_A + C_B
    eigvals_AB, V_AB = torch.linalg.eigh(C_AB)
    eigvals_AB = torch.clamp(eigvals_AB, min=0.0)
    
    V_AB_desc = torch.flip(V_AB, dims=[1])
    
    C_A_tilde = V_AB_desc.T @ C_A @ V_AB_desc
    C_B_tilde = V_AB_desc.T @ C_B @ V_AB_desc
    
    max_k_shared = min(res[str(epsilons[0])]["k_A"] + res[str(epsilons[0])]["k_B"], C_AB.shape[0]) # Use the loosest epsilon for bounds
    
    # 1. Coarse Sweep (step size 50)
    coarse_best_ks = {str(eps): 0 for eps in epsilons}
    
    def evaluate_ks(ks):
        if ks == 0:
            return {str(eps): (res[str(eps)]["k_A"], res[str(eps)]["k_B"]) for eps in epsilons}
            
        C_EA_sub = C_A_tilde[ks:, ks:]
        C_EB_sub = C_B_tilde[ks:, ks:]
        
        evals_EA = torch.linalg.eigvalsh(C_EA_sub)
        evals_EB = torch.linalg.eigvalsh(C_EB_sub)
        
        evals_EA = torch.clamp(evals_EA, min=0.0)
        evals_EB = torch.clamp(evals_EB, min=0.0)
        
        kr_dict = {}
        for eps in epsilons:
            krA = get_k_for_fidelity(evals_EA, var_A, eps)
            krB = get_k_for_fidelity(evals_EB, var_B, eps)
            kr_dict[str(eps)] = (krA, krB)
        return kr_dict
        
    for ks in range(0, max_k_shared + 1, 50):
        kr_dict = evaluate_ks(ks)
        for eps in epsilons:
            krA, krB = kr_dict[str(eps)]
            cost = ks + krA + krB
            if cost < res[str(eps)]["optimal_cost"]:
                res[str(eps)]["optimal_cost"] = cost
                res[str(eps)]["k_shared"] = ks
                res[str(eps)]["k_res_A"] = krA
                res[str(eps)]["k_res_B"] = krB
                coarse_best_ks[str(eps)] = ks
                
    # 2. Medium-tune locally (step 10)
    medium_best_ks = {str(eps): coarse_best_ks[str(eps)] for eps in epsilons}
    for eps in epsilons:
        c_best = coarse_best_ks[str(eps)]
        for ks in range(max(0, c_best - 40), min(max_k_shared, c_best + 40) + 1, 10):
            if ks % 50 == 0: continue
            kr_dict = evaluate_ks(ks)
            krA, krB = kr_dict[str(eps)]
            cost = ks + krA + krB
            if cost < res[str(eps)]["optimal_cost"]:
                res[str(eps)]["optimal_cost"] = cost
                res[str(eps)]["k_shared"] = ks
                res[str(eps)]["k_res_A"] = krA
                res[str(eps)]["k_res_B"] = krB
                medium_best_ks[str(eps)] = ks
                
    # 3. Fine-tune locally (step 1)
    for eps in epsilons:
        m_best = medium_best_ks[str(eps)]
        for ks in range(max(0, m_best - 9), min(max_k_shared, m_best + 9) + 1, 1):
            if ks % 10 == 0: continue
            kr_dict = evaluate_ks(ks)
            krA, krB = kr_dict[str(eps)]
            cost = ks + krA + krB
            if cost < res[str(eps)]["optimal_cost"]:
                res[str(eps)]["optimal_cost"] = cost
                res[str(eps)]["k_shared"] = ks
                res[str(eps)]["k_res_A"] = krA
                res[str(eps)]["k_res_B"] = krB
                
    # 4. Overlap
    overlap = get_subspace_overlap(C_A, C_B, res["0.01"]["k_A"], res["0.01"]["k_B"])
    for eps in epsilons:
        res[str(eps)]["overlap"] = overlap
        
    return res

def main():
    os.makedirs("experiments/experiment10/results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    model_id = "allenai/OLMoE-1B-7B-0924"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    layer8_experts = model.model.layers[8].mlp.experts
    
    print("Loading datasets and controls...")
    exp4_data = load_all()
    oracle_matrix = exp4_data["D_oracle"]
    
    with open("experiments/experiment9/controls.json", "r") as f:
        controls = json.load(f)
        
    pairs = [item["pair"] for item in controls]
    rw_l2 = [item["rw_l2_symmetric"] for item in controls]
    
    # Load actual CARE capability vectors
    df = pd.read_parquet("results/exp6c/expert_vectors/EXP6C_EXPERT_CAPABILITY_VECTORS.parquet")
    layer_df = df[(df['layer_idx'] == 8) & (df['checkpoint'] == 'checkpoint_10')]
    C_hat = np.stack(layer_df['C_hat'].values)
    
    print("Computing Parameter Distances and Pair Scores...")
    param_dists = []
    care_distances = []
    oracle_scores = []
    
    for (i, j) in pairs:
        w_up_i = layer8_experts.gate_up_proj.weight.data[i] if hasattr(layer8_experts.gate_up_proj, "weight") else layer8_experts.gate_up_proj[i]
        w_down_i = layer8_experts.down_proj.weight.data[i] if hasattr(layer8_experts.down_proj, "weight") else layer8_experts.down_proj[i]
        w_up_j = layer8_experts.gate_up_proj.weight.data[j] if hasattr(layer8_experts.gate_up_proj, "weight") else layer8_experts.gate_up_proj[j]
        w_down_j = layer8_experts.down_proj.weight.data[j] if hasattr(layer8_experts.down_proj, "weight") else layer8_experts.down_proj[j]
        
        dist = torch.nn.functional.mse_loss(w_up_i.float(), w_up_j.float()).item() + torch.nn.functional.mse_loss(w_down_i.float(), w_down_j.float()).item()
        param_dists.append(dist)
        
        care_distances.append(np.linalg.norm(C_hat[i] - C_hat[j]))
        oracle_scores.append(oracle_matrix[i, j])
        
    def get_top_100(scores, reverse=False):
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=reverse)
        return [x[0] for x in indexed[:100]]
        
    idx_care = get_top_100(care_distances, reverse=False)
    idx_oracle = get_top_100(oracle_scores, reverse=False)
    idx_rw = get_top_100(rw_l2, reverse=False)
    idx_param = get_top_100(param_dists, reverse=False)
    
    idx_random = list(range(len(pairs)))
    random.seed(42)
    random.shuffle(idx_random)
    idx_random = idx_random[:100]
    
    group_indices = {
        "CARE": idx_care,
        "RW-L2": idx_rw,
        "Parameter": idx_param,
        "Random": idx_random,
        "Oracle-best": idx_oracle
    }
    
    print("Computing Output Covariances on Split A...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = 512
    max_tokens = 256 * 1024
    tokens = encodings.input_ids[0, :max_tokens]
    
    input_ids = tokens.view(-1, seq_len)
    batch_size = 4
    n_batches = len(input_ids) // batch_size
    
    expert_sums = torch.zeros((64, 2048), device="cpu", dtype=torch.float64)
    expert_covs = torch.zeros((64, 2048, 2048), device="cpu", dtype=torch.float64)
    total_tokens = len(input_ids) * seq_len
    
    for b in tqdm(range(n_batches)):
        batch_ids = input_ids[b*batch_size:(b+1)*batch_size].to(device)
        
        captured_inputs = []
        class StopForward(Exception): pass
        
        def hook(module, args, kwargs):
            captured_inputs.append(args[0].detach())
            raise StopForward()
            
        layer_module = model.model.layers[8].mlp
        handle = layer_module.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            _ = model(batch_ids)
        except StopForward:
            pass
        finally:
            handle.remove()
        
        hidden = captured_inputs[0]
        
        B, S, D = hidden.shape
        h_flat = hidden.view(B * S, D)
        
        for e in range(64):
            weight_gate_up = model.model.layers[8].mlp.experts.gate_up_proj[e]
            weight_down = model.model.layers[8].mlp.experts.down_proj[e]
            
            gate_up = torch.nn.functional.linear(h_flat, weight_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            out = torch.nn.functional.linear(act, weight_down).float()
            
            expert_sums[e] += out.sum(dim=0).cpu().double()
            expert_covs[e] += (out.T @ out).cpu().double()
            
    results = {}
    
    for group_name, indices in group_indices.items():
        print(f"\nProcessing {group_name}...")
        results[group_name] = []
        
        for idx in tqdm(indices):
            i, j = pairs[idx]
            
            C_A_raw = expert_covs[i].to(device).float()
            C_B_raw = expert_covs[j].to(device).float()
            
            mu_A = (expert_sums[i] / total_tokens).to(device).float()
            mu_B = (expert_sums[j] / total_tokens).to(device).float()
            
            C_A_centered = C_A_raw - total_tokens * torch.outer(mu_A, mu_A)
            C_B_centered = C_B_raw - total_tokens * torch.outer(mu_B, mu_B)
            
            var_A = torch.diag(C_A_centered) / total_tokens
            var_B = torch.diag(C_B_centered) / total_tokens
            
            std_A = torch.sqrt(torch.clamp(var_A, min=1e-8))
            std_B = torch.sqrt(torch.clamp(var_B, min=1e-8))
            
            inv_std_A = torch.diag(1.0 / std_A)
            inv_std_B = torch.diag(1.0 / std_B)
            
            C_A_norm = inv_std_A @ C_A_centered @ inv_std_A
            C_B_norm = inv_std_B @ C_B_centered @ inv_std_B
            
            epsilons = [0.05, 0.01, 0.001]
            raw_res = compute_frc_all_eps(C_A_raw, C_B_raw, epsilons)
            norm_res = compute_frc_all_eps(C_A_norm, C_B_norm, epsilons)
            
            pair_result = {
                "pair": [int(i), int(j)],
                "raw": raw_res,
                "norm": norm_res
            }
                
            results[group_name].append(pair_result)
            
    with open("experiments/experiment10/results/frc_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Done! Results saved to experiments/experiment10/results/frc_results.json")

if __name__ == "__main__":
    main()
