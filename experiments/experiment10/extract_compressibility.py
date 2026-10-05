import torch
import numpy as np
import json
import os
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from experiments.experiment4.data_loader import load_all

torch.set_grad_enabled(False)

def get_principal_angles(C_A, C_B, k_A, k_B):
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

def compute_frc(C_A, C_B, epsilon):
    # Independent costs
    eigvals_A, _ = torch.linalg.eigh(C_A)
    eigvals_B, _ = torch.linalg.eigh(C_B)
    
    eigvals_A = torch.clamp(eigvals_A, min=0.0)
    eigvals_B = torch.clamp(eigvals_B, min=0.0)
    
    var_A = eigvals_A.sum().item()
    var_B = eigvals_B.sum().item()
    
    if var_A == 0 or var_B == 0:
        return {"k_A": 0, "k_B": 0, "optimal_cost": 0, "k_shared": 0, "k_res_A": 0, "k_res_B": 0, "overlap": 0.0}
        
    k_A = get_k_for_fidelity(eigvals_A, var_A, epsilon)
    k_B = get_k_for_fidelity(eigvals_B, var_B, epsilon)
    
    # Joint SVD
    C_AB = C_A + C_B
    eigvals_AB, V_AB = torch.linalg.eigh(C_AB)
    eigvals_AB = torch.clamp(eigvals_AB, min=0.0)
    
    V_AB_desc = torch.flip(V_AB, dims=[1])
    
    optimal_cost = float('inf')
    best_ks = 0
    best_krA = 0
    best_krB = 0
    
    max_k_shared = min(k_A + k_B, C_AB.shape[0])
    I = torch.eye(C_AB.shape[0], device=C_AB.device)
    
    for ks in range(0, max_k_shared + 1):
        if ks == 0:
            krA = k_A
            krB = k_B
        else:
            Vs = V_AB_desc[:, :ks]
            Proj = I - Vs @ Vs.T
            
            C_EA = Proj @ C_A @ Proj
            C_EB = Proj @ C_B @ Proj
            
            evals_EA = torch.linalg.eigvalsh(C_EA)
            evals_EB = torch.linalg.eigvalsh(C_EB)
            
            evals_EA = torch.clamp(evals_EA, min=0.0)
            evals_EB = torch.clamp(evals_EB, min=0.0)
            
            krA = get_k_for_fidelity(evals_EA, var_A, epsilon)
            krB = get_k_for_fidelity(evals_EB, var_B, epsilon)
            
        cost = ks + krA + krB
        if cost < optimal_cost:
            optimal_cost = cost
            best_ks = ks
            best_krA = krA
            best_krB = krB
            
    k_A_99 = get_k_for_fidelity(eigvals_A, var_A, 0.01)
    k_B_99 = get_k_for_fidelity(eigvals_B, var_B, 0.01)
    overlap = get_principal_angles(C_A, C_B, k_A_99, k_B_99)
            
    return {
        "k_A": k_A,
        "k_B": k_B,
        "optimal_cost": optimal_cost,
        "k_shared": best_ks,
        "k_res_A": best_krA,
        "k_res_B": best_krB,
        "overlap": overlap
    }

def main():
    os.makedirs("experiments/experiment10/results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model for expert weights...")
    model_id = "allenai/OLMoE-1B-7B-0924"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    layer8_experts = model.model.layers[8].mlp.experts
    
    print("Loading datasets and controls...")
    exp4_data = load_all()
    oracle_matrix = exp4_data["oracle_matrix"]
    
    with open("experiments/experiment9/controls.json", "r") as f:
        controls = json.load(f)
        
    pairs = controls["pairs"]
    rw_l2 = controls["rw_l2_symmetric"]
    
    print("Computing Parameter Distances...")
    param_dists = []
    for (i, j) in pairs:
        w_up_i = layer8_experts[i].gate_up_proj.weight.data
        w_down_i = layer8_experts[i].down_proj.weight.data
        w_up_j = layer8_experts[j].gate_up_proj.weight.data
        w_down_j = layer8_experts[j].down_proj.weight.data
        
        dist = torch.nn.functional.mse_loss(w_up_i, w_up_j).item() + torch.nn.functional.mse_loss(w_down_i, w_down_j).item()
        param_dists.append(dist)
        
    del model
    torch.cuda.empty_cache()
    
    def get_top_100(scores, reverse=False):
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=reverse)
        return [x[0] for x in indexed[:100]]
        
    care_scores = []
    for (i, j) in pairs:
        care_scores.append(oracle_matrix[i, j])
        
    idx_care = get_top_100(care_scores, reverse=False)
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
        "Random": idx_random
    }
    
    print("Computing Output Covariances on Split A...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device)
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
        
        hidden = model.model.embed_tokens(batch_ids)
        for i in range(8):
            hidden = model.model.layers[i](hidden)[0]
            
        hidden = model.model.layers[8].input_layernorm(hidden)
        
        B, S, D = hidden.shape
        h_flat = hidden.view(B * S, D)
        
        for e in range(64):
            expert = model.model.layers[8].mlp.experts[e]
            gate_up = expert.gate_up_proj(h_flat)
            gate, up = gate_up.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            out = expert.down_proj(act)
            
            expert_sums[e] += out.sum(dim=0).cpu().double()
            expert_covs[e] += (out.T @ out).cpu().double()
            
    results = {}
    
    for group_name, indices in group_indices.items():
        print(f"\nProcessing {group_name}...")
        results[group_name] = []
        
        for idx in tqdm(indices):
            i, j = pairs[idx]
            
            C_A_raw = expert_covs[i].to(device)
            C_B_raw = expert_covs[j].to(device)
            
            mu_A = (expert_sums[i] / total_tokens).to(device)
            mu_B = (expert_sums[j] / total_tokens).to(device)
            
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
            
            pair_result = {
                "pair": [i, j],
                "raw": {},
                "norm": {}
            }
            
            for eps in [0.05, 0.01, 0.001]:
                pair_result["raw"][str(eps)] = compute_frc(C_A_raw, C_B_raw, eps)
                pair_result["norm"][str(eps)] = compute_frc(C_A_norm, C_B_norm, eps)
                
            results[group_name].append(pair_result)
            
    with open("experiments/experiment10/results/frc_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Done! Results saved to experiments/experiment10/results/frc_results.json")

if __name__ == "__main__":
    main()
