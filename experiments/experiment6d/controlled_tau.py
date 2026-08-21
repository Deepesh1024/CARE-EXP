"""
EXPERIMENT 6D - CONTROLLED TAU
============================================================
Selects controlled experts spanning the magnitude quantiles and
constructs precisely realizable token environments (tau) along
the feasible angular arc up to theta_max.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs, ALPHAS, TARGET_ANGLES_DEG, CHECKPOINT_NAME

def get_selected_experts(df_c):
    # Select early (2), middle (8), late (14)
    target_layers = [2, 8, 14]
    df = df_c[df_c["layer_idx"].isin(target_layers)].copy()
    
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    selected = []
    
    for l in target_layers:
        df_l = df[df["layer_idx"] == l]
        mags = df_l["C_mag"].values
        
        for q in quantiles:
            target_mag = np.quantile(mags, q)
            # Find the 2 experts closest to this quantile magnitude
            df_l = df_l.assign(dist=np.abs(df_l["C_mag"] - target_mag))
            closest = df_l.nsmallest(2, "dist")
            
            for _, row in closest.iterrows():
                selected.append({
                    "layer_idx": row["layer_idx"],
                    "expert_idx": row["expert_idx"],
                    "quantile": q,
                    "C_i": row["C_raw"]
                })
                
    return pd.DataFrame(selected)

def construct_targets(df_experts):
    targets = []
    
    for _, row in df_experts.iterrows():
        c = np.array(row["C_i"])
        c = np.maximum(c, 1e-12)
        mag_c = np.linalg.norm(c)
        c_hat = c / mag_c
        
        # Calculate theta_max
        min_idx = np.argmin(c)
        theta_max_rad = np.arccos(c[min_idx] / mag_c)
        theta_max_deg = np.degrees(theta_max_rad)
        
        # Orthogonal direction along the e_k ray (guarantees positive orthant realizability)
        e_k = np.zeros(10)
        e_k[min_idx] = 1.0
        u = e_k - np.dot(e_k, c_hat) * c_hat
        u_hat = u / np.linalg.norm(u)
        
        angles = TARGET_ANGLES_DEG + [theta_max_deg]
        
        for alpha in ALPHAS:
            for theta_deg in angles:
                if theta_deg > theta_max_deg:
                    continue # Skip infeasible conditions explicitly
                    
                theta_rad = np.radians(theta_deg)
                
                # Construct tau
                tau_hat = np.cos(theta_rad) * c_hat + np.sin(theta_rad) * u_hat
                tau_actual = alpha * tau_hat
                
                # Verify orthant constraint mathematically
                assert np.all(tau_actual >= -1e-6), "Mathematical violation of positive orthant!"
                tau_actual = np.maximum(tau_actual, 0.0) # clip float precision
                
                targets.append({
                    "layer_idx": row["layer_idx"],
                    "expert_idx": row["expert_idx"],
                    "quantile": row["quantile"],
                    "alpha": alpha,
                    "target_angle_deg": float(theta_deg),
                    "is_theta_max": (theta_deg == theta_max_deg),
                    "tau_actual": tau_actual.tolist(),
                    "tau_hat": tau_hat.tolist(),
                    "mag_tau_actual": np.linalg.norm(tau_actual)
                })
                
    return pd.DataFrame(targets)

def main():
    ensure_dirs()
    print("Loading 6C Functional States...")
    c_path = os.path.join(DIRS["exp6c_expert_vectors"], "EXP6C_EXPERT_CAPABILITY_VECTORS.parquet")
    df_c = pd.read_parquet(c_path)
    df_c = df_c[df_c["checkpoint"] == CHECKPOINT_NAME]
    
    print("Selecting Controlled Experts...")
    df_experts = get_selected_experts(df_c)
    
    print("Constructing Realizable Tau Environments...")
    df_targets = construct_targets(df_experts)
    
    out_path = os.path.join(DIRS["results"], "EXP6D_TAU_ACTUAL.parquet")
    df_targets.to_parquet(out_path)
    print(f"Saved {len(df_targets)} tau environments to {out_path}")

if __name__ == "__main__":
    main()
