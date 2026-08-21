"""
EXPERIMENT 6D - ANALYSIS
============================================================
Evaluates the primary and secondary hypotheses of the controlled
intervention, comparing linear and nonlinear fits.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / (ss_tot + 1e-12))

def fit_models(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    # 1. Linear: y = a*x + b
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    y_lin = lr.predict(x.reshape(-1, 1))
    r2_lin = r2(y, y_lin)
    
    return {
        "linear_r2": float(r2_lin),
    }

def main():
    ensure_dirs()
    print("Running Final Analysis...")
    
    in_path = os.path.join(DIRS["results"], "EXP6D_GEOMETRY.parquet")
    if not os.path.exists(in_path):
        print(f"Skipping: {in_path} not found.")
        return
        
    df = pd.read_parquet(in_path)
    # Filter out controls for primary analysis
    df_primary = df[(df["target_angle_deg"] >= 0) & (df["seed"] == 42)]
    
    stats = {}
    
    # --- 11. PRIMARY TEST ---
    print("Testing Primary Hypotheses (Delta_theta)")
    y = df_primary["delta_theta"].values
    x1 = df_primary["mag_tau"].values
    x2 = df_primary["mag_tau_perp"].values
    x3 = df_primary["susceptibility_ratio"].values
    
    stats["Primary_Test"] = {
        "Y_vs_X1_mag_tau": fit_models(x1, y),
        "Y_vs_X2_mag_tau_perp": fit_models(x2, y),
        "Y_vs_X3_susceptibility_ratio": fit_models(x3, y)
    }
    
    # --- 12. SECONDARY TESTS ---
    print("Testing Secondary Hypotheses")
    y_dc_perp = df_primary["mag_DeltaC_perp"].values
    stats["Secondary_Test_DeltaC_perp"] = fit_models(x3, y_dc_perp)
    
    y_dc_par = df_primary["mag_DeltaC_par"].values
    x_tau_par = df_primary["mag_tau_par"].values
    stats["Secondary_Test_DeltaC_par"] = fit_models(x_tau_par, y_dc_par)
    
    # Aggregate angles
    stats["Angles"] = {
        "Mean_Angle_C_tau": float(df_primary["angle_c_tau"].mean()),
        "Mean_Angle_tau_DeltaC": float(df_primary["angle_tau_dc"].mean()),
        "Mean_Angle_C_DeltaC": float(df_primary["angle_c_dc"].mean())
    }
    
    out_path = os.path.join(DIRS["results"], "EXP6D_STATISTICS.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Saved statistics to {out_path}")

if __name__ == "__main__":
    main()
