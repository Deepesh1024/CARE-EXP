"""
EXPERIMENT 6C - PHASE 5: STATE-DEPENDENT DYNAMICS MODELING
============================================================
Tests the core interaction hypothesis: DeltaC_i = F(C_i, tau_i, I_i).
Evaluates 10D directional alignments (cosines) and linear models.
Computes functional directional stability (Susceptibility) and runs 
sensitivity analyses for angular stability.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, TRANSITIONS, NUM_LAYERS, ensure_dirs

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

def run_directional_dynamics():
    print("Running directional dynamics modeling...")
    df_mov = pd.read_parquet(os.path.join(DIRS["trajectories"], "EXP6C_EXPERT_MOVEMENT.parquet"))
    
    results = {}
    
    for trans in df_mov["transition"].unique():
        trans_data = df_mov[df_mov["transition"] == trans].copy()
        
        # 1. 10D Directional Alignments
        cos_tau = []
        cos_c = []
        cos_i = []
        cos_perp = []
        
        X_tau_flat = []
        X_c_flat = []
        X_i_flat = []
        Y_flat = []
        
        X_i_perp_flat = []
        Y_perp_flat = []
        
        for idx, row in trans_data.iterrows():
            c = row["C_i_t"]
            tau = row["tau_i"]
            dc = row["DeltaC_i"]
            interaction = row["I_i"]
            
            dc_perp = row["DeltaC_perp"]
            i_perp = row["I_perp"]
            
            cos_tau.append(cosine_similarity(dc, tau))
            cos_c.append(cosine_similarity(dc, c))
            cos_i.append(cosine_similarity(dc, interaction))
            cos_perp.append(cosine_similarity(dc_perp, i_perp))
            
            for k in range(10):
                X_tau_flat.append(tau[k])
                X_c_flat.append(c[k])
                X_i_flat.append(interaction[k])
                Y_flat.append(dc[k])
                
                X_i_perp_flat.append(i_perp[k])
                Y_perp_flat.append(dc_perp[k])
                
        trans_data["cos_DeltaC_tau"] = cos_tau
        trans_data["cos_DeltaC_C"] = cos_c
        trans_data["cos_DeltaC_I"] = cos_i
        trans_data["cos_DeltaC_perp_I_perp"] = cos_perp
        
        # Means
        mean_cos_tau = np.mean(cos_tau)
        mean_cos_c = np.mean(cos_c)
        mean_cos_i = np.mean(cos_i)
        mean_cos_perp = np.mean(cos_perp)
        
        # 2. Predictive Models (Flattened over 10D axes)
        X_tau_flat = np.array(X_tau_flat).reshape(-1, 1)
        X_c_flat = np.array(X_c_flat).reshape(-1, 1)
        X_i_flat = np.array(X_i_flat).reshape(-1, 1)
        Y_flat = np.array(Y_flat)
        
        X_i_perp_flat = np.array(X_i_perp_flat).reshape(-1, 1)
        Y_perp_flat = np.array(Y_perp_flat)
        
        m_tau = LinearRegression().fit(X_tau_flat, Y_flat)
        m_c = LinearRegression().fit(X_c_flat, Y_flat)
        
        X_c_tau = np.hstack([X_c_flat, X_tau_flat])
        m_c_tau = LinearRegression().fit(X_c_tau, Y_flat)
        
        m_i = LinearRegression().fit(X_i_flat, Y_flat)
        
        X_full = np.hstack([X_c_flat, X_tau_flat, X_i_flat])
        m_full = LinearRegression().fit(X_full, Y_flat)
        
        m_perp = LinearRegression().fit(X_i_perp_flat, Y_perp_flat)
        
        # 3. Angular Susceptibility and Stability Sensitivity
        # S_theta = |theta_10D| / (||tau|| + epsilon)
        epsilon = 1e-6
        trans_data["S_theta"] = trans_data["theta_10D"] / (trans_data["mag_tau"] + epsilon)
        
        # Filter thresholds per layer
        sensitivity_results = {}
        for pct in [5, 10, 20]:
            valid_experts = []
            for l in range(NUM_LAYERS):
                layer_data = trans_data[trans_data["layer_idx"] == l]
                threshold = np.percentile(layer_data["mag_C"], pct)
                valid_experts.append(layer_data[layer_data["mag_C"] >= threshold])
                
            filtered_df = pd.concat(valid_experts)
            
            # Test if layer affects S_theta
            # Categorize layer: 0-4 early, 5-10 middle, 11-15 late
            filtered_df["layer_group"] = pd.cut(filtered_df["layer_idx"], bins=[-1, 4, 10, 16], labels=["early", "middle", "late"])
            
            # Dummy encoding
            filtered_df = pd.get_dummies(filtered_df, columns=["layer_group"], drop_first=True)
            
            # Regression: S_theta ~ mag_C + mag_tau + layer_middle + layer_late
            X_cols = ["mag_C", "mag_tau"]
            if "layer_group_middle" in filtered_df.columns: X_cols.append("layer_group_middle")
            if "layer_group_late" in filtered_df.columns: X_cols.append("layer_group_late")
            
            X_sus = filtered_df[X_cols]
            X_sus = sm.add_constant(X_sus)
            Y_sus = filtered_df["S_theta"]
            
            model = sm.OLS(Y_sus, X_sus.astype(float)).fit()
            
            sensitivity_results[f"pct_{pct}"] = {
                "threshold_used": f"{pct}%",
                "experts_retained": len(filtered_df),
                "r2": model.rsquared,
                "pvalues": model.pvalues.to_dict()
            }
            
        results[trans] = {
            "Mean_Cos_DeltaC_Tau": float(mean_cos_tau),
            "Mean_Cos_DeltaC_C": float(mean_cos_c),
            "Mean_Cos_DeltaC_I": float(mean_cos_i),
            "Mean_Cos_DeltaC_Perp_I_Perp": float(mean_cos_perp),
            "R2_Tau_Only": r2_score(Y_flat, m_tau.predict(X_tau_flat)),
            "R2_C_Only": r2_score(Y_flat, m_c.predict(X_c_flat)),
            "R2_C_plus_Tau": r2_score(Y_flat, m_c_tau.predict(X_c_tau)),
            "R2_I_Only": r2_score(Y_flat, m_i.predict(X_i_flat)),
            "R2_C_plus_Tau_plus_I": r2_score(Y_flat, m_full.predict(X_full)),
            "R2_I_Perp_predicting_DeltaC_Perp": r2_score(Y_perp_flat, m_perp.predict(X_i_perp_flat)),
            "Angular_Susceptibility_Sensitivity": sensitivity_results
        }
        
    out_path = os.path.join(DIRS["models"], "EXP6C_MODEL_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved {out_path}")

def main():
    ensure_dirs()
    run_directional_dynamics()

if __name__ == "__main__":
    main()
