"""
EXPERIMENT 6C - PHASE 7: PLOTTING
============================================================
Generates the required visualization figures for directional 
dynamics, susceptibility, and alignments.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs

sns.set_theme(style="darkgrid", context="talk")

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

def generate_plots():
    print("Generating directional dynamics plots...")
    mov_path = os.path.join(DIRS["trajectories"], "EXP6C_EXPERT_MOVEMENT.parquet")
    if not os.path.exists(mov_path):
        print(f"Skipping plots: {mov_path} not found.")
        return
        
    df = pd.read_parquet(mov_path)
    
    # Use only the first transition for clean plotting, or facet across them
    trans = df["transition"].unique()[0]
    df_t = df[df["transition"] == trans].copy()
    
    # Calculate cosine alignments for plot 5
    cos_c = []
    cos_tau = []
    cos_i = []
    for idx, row in df_t.iterrows():
        c = np.array(row["C_i_t"])
        tau = np.array(row["tau_i"])
        dc = np.array(row["DeltaC_i"])
        interaction = c * tau
        
        cos_c.append(cosine_similarity(dc, c))
        cos_tau.append(cosine_similarity(dc, tau))
        cos_i.append(cosine_similarity(dc, interaction))
        
    df_t["cos_DeltaC_C"] = cos_c
    df_t["cos_DeltaC_tau"] = cos_tau
    df_t["cos_DeltaC_I"] = cos_i
    
    # 1. ||C|| vs |Delta_theta_C|
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_t, x="mag_C", y=np.abs(df_t["Delta_theta_C_2d"]), alpha=0.6)
    plt.title("State Magnitude ||C|| vs Absolute Angular Shift |Δθ_C|")
    plt.xlabel("State Magnitude ||C||")
    plt.ylabel("Absolute Angular Shift |Δθ_C| (radians)")
    plt.savefig(os.path.join(DIRS["plots"], "1_magC_vs_angular_shift.png"), bbox_inches='tight')
    plt.close()
    
    # 2. ||tau|| vs |Delta_theta_C|
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_t, x="mag_tau", y=np.abs(df_t["Delta_theta_C_2d"]), alpha=0.6)
    plt.title("Environment Magnitude ||τ|| vs Absolute Angular Shift |Δθ_C|")
    plt.xlabel("Environment Magnitude ||τ||")
    plt.ylabel("Absolute Angular Shift |Δθ_C| (radians)")
    plt.savefig(os.path.join(DIRS["plots"], "2_magTau_vs_angular_shift.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Angular separation(theta_tau, theta_C) vs Delta_theta_C
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_t, x="Delta_theta_tau_C_2d", y="Delta_theta_C_2d", alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Angular Separation (θ_τ - θ_C) vs Angular Shift Δθ_C")
    plt.xlabel("Angular Separation (θ_τ - θ_C)")
    plt.ylabel("Angular Shift Δθ_C")
    plt.savefig(os.path.join(DIRS["plots"], "3_separation_vs_shift.png"), bbox_inches='tight')
    plt.close()
    
    # 4. Layer vs Directional Susceptibility
    # Filter bottom 10% magnitude experts to prevent asymptote
    threshold = np.percentile(df_t["mag_C"], 10)
    df_filtered = df_t[df_t["mag_C"] >= threshold].copy()
    epsilon = 1e-6
    df_filtered["S_theta"] = np.abs(df_filtered["Delta_theta_C_2d"]) / (df_filtered["mag_tau"] + epsilon)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x="layer_idx", y="S_theta", color="lightblue")
    plt.title("Directional Susceptibility S_θ across Layers (Bottom 10% ||C|| excluded)")
    plt.xlabel("Layer Index")
    plt.ylabel("Susceptibility S_θ = |Δθ_C| / ||τ||")
    plt.yscale("log")
    plt.savefig(os.path.join(DIRS["plots"], "4_layer_vs_susceptibility.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Cosine Alignments (10D) KDE Distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_t, x="cos_DeltaC_C", label="cos(ΔC, C)", fill=True, alpha=0.3)
    sns.kdeplot(data=df_t, x="cos_DeltaC_tau", label="cos(ΔC, τ)", fill=True, alpha=0.3)
    sns.kdeplot(data=df_t, x="cos_DeltaC_I", label="cos(ΔC, C ⊙ τ)", fill=True, alpha=0.3)
    plt.title("10D Directional Alignment Distributions")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(DIRS["plots"], "5_cosine_alignments_10D.png"), bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {DIRS['plots']}")

def main():
    ensure_dirs()
    generate_plots()

if __name__ == "__main__":
    main()
