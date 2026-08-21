"""
EXPERIMENT 6D - CONTROLS
============================================================
Implements the statistical verification of the controls:
- Baseline & Random drift comparison
- Magnitude vs Direction ablation permutations
- Replication variance (Seed consistency)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs

def get_r2(x, y):
    if len(x) == 0: return 0.0
    model = LinearRegression()
    model.fit(x, y)
    return r2_score(y, model.predict(x))

def main():
    ensure_dirs()
    print("Running Controls Analysis...")
    
    in_path = os.path.join(DIRS["results"], "EXP6D_GEOMETRY.parquet")
    if not os.path.exists(in_path):
        print(f"Skipping: {in_path} not found.")
        return
        
    df = pd.read_parquet(in_path)
    
    # 1. Control A & B: Baseline and Random Drift
    # target_angle_deg = -1 (Baseline), -2 (Random)
    df_baseline = df[df["target_angle_deg"] == -1.0]
    df_random = df[df["target_angle_deg"] == -2.0]
    df_real = df[(df["target_angle_deg"] >= 0) & (df["seed"] == 42)] # Primary seed
    
    baseline_drift = df_baseline["delta_theta"].mean() if len(df_baseline) > 0 else 0.0
    random_drift = df_random["delta_theta"].mean() if len(df_random) > 0 else 0.0
    real_drift = df_real["delta_theta"].mean() if len(df_real) > 0 else 0.0
    
    # 2. Control C & D: Permutation Nulls for Primary Hypothesis
    y = df_real["delta_theta"].values
    x_true = df_real["susceptibility_ratio"].values.reshape(-1, 1)
    true_r2 = get_r2(x_true, y)
    
    # Null C: Same Magnitude, Different Direction (Permute direction)
    # Null D: Same Direction, Different Magnitude (Permute magnitude)
    N_PERM = 100
    null_c_r2s = []
    null_d_r2s = []
    
    mag_tau = df_real["mag_tau"].values
    tau_perp = df_real["mag_tau_perp"].values
    c_mag = df_real["mag_C"].values + 1e-12
    
    for _ in range(N_PERM):
        # Null C: Keep magnitude constant, shuffle direction (tau_perp/mag_tau is the directional component)
        shuffled_dir = np.random.permutation(tau_perp / (mag_tau + 1e-12))
        x_null_c = (mag_tau * shuffled_dir) / c_mag
        null_c_r2s.append(get_r2(x_null_c.reshape(-1, 1), y))
        
        # Null D: Keep direction constant, shuffle magnitude
        shuffled_mag = np.random.permutation(mag_tau)
        dir_comp = tau_perp / (mag_tau + 1e-12)
        x_null_d = (shuffled_mag * dir_comp) / c_mag
        null_d_r2s.append(get_r2(x_null_d.reshape(-1, 1), y))
        
    mean_null_c = np.mean(null_c_r2s)
    z_c = (true_r2 - mean_null_c) / (np.std(null_c_r2s) + 1e-12)
    
    mean_null_d = np.mean(null_d_r2s)
    z_d = (true_r2 - mean_null_d) / (np.std(null_d_r2s) + 1e-12)
    
    # 3. Control E: Seed Variance
    # We find conditions that were run with multiple seeds and calculate the std dev of Delta_theta
    df_reps = df[df.duplicated(subset=["layer_idx", "expert_idx", "alpha", "target_angle_deg"], keep=False)]
    if len(df_reps) > 0:
        seed_variance = df_reps.groupby(["layer_idx", "expert_idx", "alpha", "target_angle_deg"])["delta_theta"].std().mean()
    else:
        seed_variance = 0.0
        
    results = {
        "Control_A_Baseline_Drift_deg": baseline_drift,
        "Control_B_Random_Drift_deg": random_drift,
        "Mean_Real_Response_deg": real_drift,
        "True_R2": true_r2,
        "Null_C_Z_Score": z_c,
        "Null_D_Z_Score": z_d,
        "Control_E_Mean_Seed_Std_deg": seed_variance
    }
    
    out_path = os.path.join(DIRS["results"], "EXP6D_CONTROL_RESULTS.parquet")
    pd.DataFrame([results]).to_parquet(out_path)
    print(f"Saved controls to {out_path}")

if __name__ == "__main__":
    main()
