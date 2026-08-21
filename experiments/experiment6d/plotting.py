"""
EXPERIMENT 6D - PLOTTING
============================================================
Generates the 10 requested geometric and response curve plots.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs

def main():
    ensure_dirs()
    print("Running Final Plotting...")
    
    in_path = os.path.join(DIRS["results"], "EXP6D_GEOMETRY.parquet")
    if not os.path.exists(in_path):
        print(f"Skipping: {in_path} not found.")
        return
        
    df = pd.read_parquet(in_path)
    # Filter out controls for plotting
    df = df[(df["target_angle_deg"] >= 0) & (df["seed"] == 42)]
    
    plots_dir = os.path.join(DIRS["results"], "plots")
    
    sns.set_theme(style="whitegrid")
    
    # 1. ||tau|| vs Delta_theta
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mag_tau", y="delta_theta", hue="target_angle_deg", palette="viridis")
    plt.title("1. ||tau|| vs Delta_theta")
    plt.savefig(os.path.join(plots_dir, "01_tau_vs_delta_theta.png"))
    plt.close()
    
    # 2. ||tau_perpendicular|| vs Delta_theta
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mag_tau_perp", y="delta_theta", hue="target_angle_deg", palette="viridis")
    plt.title("2. ||tau_perp|| vs Delta_theta")
    plt.savefig(os.path.join(plots_dir, "02_tau_perp_vs_delta_theta.png"))
    plt.close()
    
    # 3 & 10. ||tau_perpendicular|| / ||C|| vs Delta_theta (The Collapse Plot)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="susceptibility_ratio", y="delta_theta", hue="quantile", palette="coolwarm")
    plt.title("3/10. Collapse Plot: Susceptibility Ratio vs Delta_theta")
    plt.xlabel("||tau_perp|| / ||C||")
    plt.savefig(os.path.join(plots_dir, "03_10_collapse_plot.png"))
    plt.close()
    
    # 4. ||C|| vs Delta_theta at fixed ||tau|| (e.g., alpha=1.0)
    df_fixed = df[df["alpha"] == 1.0]
    if len(df_fixed) > 0:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_fixed, x="mag_C", y="delta_theta", hue="target_angle_deg", palette="viridis")
        plt.title("4. ||C|| vs Delta_theta (Fixed alpha=1.0)")
        plt.savefig(os.path.join(plots_dir, "04_C_vs_delta_theta_fixed_tau.png"))
        plt.close()
        
    # 5. Angle(C,tau) vs Delta_theta
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="angle_c_tau", y="delta_theta", hue="alpha", palette="crest")
    plt.title("5. Angle(C, tau) vs Delta_theta")
    plt.savefig(os.path.join(plots_dir, "05_angle_vs_delta_theta.png"))
    plt.close()
    
    # 6. ||tau|| vs ||DeltaC_perpendicular||
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mag_tau", y="mag_DeltaC_perp", hue="target_angle_deg", palette="viridis")
    plt.title("6. ||tau|| vs ||DeltaC_perp||")
    plt.savefig(os.path.join(plots_dir, "06_tau_vs_deltac_perp.png"))
    plt.close()
    
    # 7. ||tau|| vs ||DeltaC_parallel||
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mag_tau", y="mag_DeltaC_par", hue="target_angle_deg", palette="viridis")
    plt.title("7. ||tau|| vs ||DeltaC_par||")
    plt.savefig(os.path.join(plots_dir, "07_tau_vs_deltac_par.png"))
    plt.close()
    
    # 8. Delta_theta curves for same / intermediate / orthogonal tau directions
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="alpha", y="delta_theta", hue="target_angle_deg", palette="viridis")
    plt.title("8. Delta_theta curves by Target Angle")
    plt.savefig(os.path.join(plots_dir, "08_delta_theta_curves_by_angle.png"))
    plt.close()
    
    # 9. Directional response curves grouped by ||C|| quantile
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="alpha", y="delta_theta", hue="quantile", palette="coolwarm")
    plt.title("9. Directional Response grouped by ||C|| Quantile")
    plt.savefig(os.path.join(plots_dir, "09_directional_response_by_quantile.png"))
    plt.close()
    
    print(f"Saved 10 plots to {plots_dir}")

if __name__ == "__main__":
    main()
