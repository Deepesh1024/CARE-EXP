"""
EXPERIMENT 6C - PHASE 6: NULL CONTROLS
============================================================
Evaluates directional alignments against random-direction 
and permutation nulls.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, NUM_LAYERS, ensure_dirs

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

def run_null_controls():
    print("Running Null Controls (Random Direction & Permutation)...")
    df_mov = pd.read_parquet(os.path.join(DIRS["trajectories"], "EXP6C_EXPERT_MOVEMENT.parquet"))
    
    results = {}
    
    for trans in df_mov["transition"].unique():
        trans_data = df_mov[df_mov["transition"] == trans]
        
        obs_cos_i = []
        obs_cos_i_perp = []
        for idx, row in trans_data.iterrows():
            c = row["C_i_t"]
            tau = row["tau_i"]
            dc = row["DeltaC_i"]
            interaction = row["I_i"]
            dc_perp = row["DeltaC_perp"]
            i_perp = row["I_perp"]
            
            obs_cos_i.append(cosine_similarity(dc, interaction))
            obs_cos_i_perp.append(cosine_similarity(dc_perp, i_perp))
        
        obs_mean_i = np.mean(obs_cos_i)
        obs_mean_i_perp = np.mean(obs_cos_i_perp)
        
        # 1. Random Direction Null
        # Draw random 10D vectors from standard normal, which are spherically symmetric
        null_random_cos_i = []
        null_random_cos_i_perp = []
        for idx, row in trans_data.iterrows():
            dc = row["DeltaC_i"]
            dc_perp = row["DeltaC_perp"]
            v_rand = np.random.randn(10)
            null_random_cos_i.append(cosine_similarity(dc, v_rand))
            null_random_cos_i_perp.append(cosine_similarity(dc_perp, v_rand))
            
        null_rand_mean = np.mean(null_random_cos_i)
        null_rand_std = np.std(null_random_cos_i)
        
        null_rand_mean_perp = np.mean(null_random_cos_i_perp)
        null_rand_std_perp = np.std(null_random_cos_i_perp)
        
        # 2. Permutation Null
        # Shuffle tau within each layer
        null_perm_cos_i = []
        null_perm_cos_i_perp = []
        for l in range(NUM_LAYERS):
            layer_data = trans_data[trans_data["layer_idx"] == l]
            taus = layer_data["tau_i"].tolist()
            np.random.shuffle(taus) # in-place permutation
            
            for i, (idx, row) in enumerate(layer_data.iterrows()):
                c = np.array(row["C_i_t"])
                dc = np.array(row["DeltaC_i"])
                dc_perp = np.array(row["DeltaC_perp"])
                tau_perm = np.array(taus[i])
                
                i_perm = c * tau_perm
                
                mag_c = np.linalg.norm(c)
                if mag_c > 0:
                    c_hat = c / mag_c
                    i_perm_par = np.dot(i_perm, c_hat) * c_hat
                    i_perm_perp = i_perm - i_perm_par
                else:
                    i_perm_perp = np.zeros(10)
                
                null_perm_cos_i.append(cosine_similarity(dc, i_perm))
                null_perm_cos_i_perp.append(cosine_similarity(dc_perp, i_perm_perp))
                
        null_perm_mean = np.mean(null_perm_cos_i)
        null_perm_std = np.std(null_perm_cos_i)
        
        null_perm_mean_perp = np.mean(null_perm_cos_i_perp)
        null_perm_std_perp = np.std(null_perm_cos_i_perp)
        
        # Z-scores
        z_rand = (obs_mean_i - null_rand_mean) / (null_rand_std / np.sqrt(len(obs_cos_i)) + 1e-9)
        z_perm = (obs_mean_i - null_perm_mean) / (null_perm_std / np.sqrt(len(obs_cos_i)) + 1e-9)
        
        z_rand_perp = (obs_mean_i_perp - null_rand_mean_perp) / (null_rand_std_perp / np.sqrt(len(obs_cos_i_perp)) + 1e-9)
        z_perm_perp = (obs_mean_i_perp - null_perm_mean_perp) / (null_perm_std_perp / np.sqrt(len(obs_cos_i_perp)) + 1e-9)
        
        results[trans] = {
            "Observed_Mean_Cos_I": float(obs_mean_i),
            "Observed_Mean_Cos_I_Perp": float(obs_mean_i_perp),
            "Random_Null_Mean": float(null_rand_mean),
            "Random_Null_Z_Score": float(z_rand),
            "Random_Null_Z_Score_Perp": float(z_rand_perp),
            "Permutation_Null_Mean": float(null_perm_mean),
            "Permutation_Null_Z_Score": float(z_perm),
            "Permutation_Null_Z_Score_Perp": float(z_perm_perp),
            "Significant_vs_Random": bool(abs(z_rand) > 3.0),
            "Significant_vs_Permutation": bool(abs(z_perm) > 3.0),
            "Perp_Significant_vs_Random": bool(abs(z_rand_perp) > 3.0),
            "Perp_Significant_vs_Permutation": bool(abs(z_perm_perp) > 3.0)
        }
        
    # 3. Spatial Convergence Null (Task-Overlap vs DeltaD)
    print("Running Spatial Convergence Null Controls (Task-Overlap vs DeltaD)...")
    df_pair = pd.read_parquet(os.path.join(DIRS["pair_metrics"], "EXP6C_PAIR_METRICS.parquet"))
    from scipy.stats import spearmanr
    
    for trans in df_pair["transition"].unique():
        pair_data = df_pair[df_pair["transition"] == trans]
        
        delta_d = pair_data["DeltaD_ij"].values
        overlap = pair_data["task_overlap"].values
        
        obs_rho, _ = spearmanr(delta_d, overlap)
        
        # Permutation Null: Shuffle overlap
        null_rhos = []
        for _ in range(50): # 50 permutations is plenty for a Z-score estimate
            overlap_perm = np.random.permutation(overlap)
            rho_perm, _ = spearmanr(delta_d, overlap_perm)
            null_rhos.append(rho_perm)
            
        null_rho_mean = np.mean(null_rhos)
        null_rho_std = np.std(null_rhos)
        
        z_score = (obs_rho - null_rho_mean) / (null_rho_std + 1e-9)
        
        if trans in results:
            results[trans]["Observed_Overlap_DeltaD_Rho"] = float(obs_rho)
            results[trans]["Z_Score"] = float(z_score)
            results[trans]["Significant"] = bool(abs(z_score) > 3.0)
        
    out_path = os.path.join(DIRS["nulls"], "EXP6C_NULL_RESULTS.json")
    
    # Merge with existing nulls from previous phase if they exist
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        for k in results.keys():
            if k in existing:
                existing[k].update(results[k])
            else:
                existing[k] = results[k]
        results = existing
        
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved {out_path}")

def main():
    ensure_dirs()
    run_null_controls()

if __name__ == "__main__":
    main()
