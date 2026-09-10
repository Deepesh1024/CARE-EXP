"""
EXPERIMENT 7A - PHASE 5: STATISTICAL ANALYSIS & REPORTING
===========================================================
Computes Spearman correlation between T_K and ΔS_K, 
enrichment metrics, and generates summary plots.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# We'll put plotting inside a try-except so it doesn't crash if matplotlib is missing
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None

from config import (
    INTERVENTION_DIR, SENSITIVITY_DIR, ANALYSIS_DIR, PLOTS_DIR,
    ensure_dirs, mark_task, is_task_completed
)

def run_analysis():
    task_id = "phase5_analysis"
    if is_task_completed(task_id):
        print("[Phase 5] Analysis already completed. Skipping.")
        return

    ensure_dirs()
    
    res_path = os.path.join(INTERVENTION_DIR, "intervention_results.csv")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Missing {res_path}. Run phase4 first.")
        
    df = pd.read_csv(res_path)
    
    # Load baselines
    a_n_scores = np.load(os.path.join(SENSITIVITY_DIR, "A_n.npy"))
    w_n_scores = np.load(os.path.join(SENSITIVITY_DIR, "W_n.npy"))
    
    # Attach baseline scores
    df['A_n'] = df.apply(lambda row: a_n_scores[int(row['expert_idx']), int(row['neuron_idx'])], axis=1)
    df['W_n'] = df.apply(lambda row: w_n_scores[int(row['expert_idx']), int(row['neuron_idx'])], axis=1)
    
    # 1. Spearman Correlations
    stats = {}
    
    for metric in ['T_K', 'A_n', 'W_n']:
        rho, pval = spearmanr(df[metric], df['delta_acc'])
        stats[f"spearman_{metric}_acc"] = {"rho": float(rho), "p_value": float(pval)}
        
        rho_p, pval_p = spearmanr(df[metric], df['delta_p_correct'])
        stats[f"spearman_{metric}_prob"] = {"rho": float(rho_p), "p_value": float(pval_p)}
        
        rho_m, pval_m = spearmanr(df[metric], df['delta_margin'])
        stats[f"spearman_{metric}_margin"] = {"rho": float(rho_m), "p_value": float(pval_m)}
        
    # 2. Top-Rank Enrichment
    # Define actual high-damage neurons: Top 20% according to delta_acc
    k_top = max(1, int(0.2 * len(df)))
    actual_top = df.nlargest(k_top, 'delta_acc').index
    
    for metric in ['T_K', 'A_n', 'W_n']:
        pred_top = df.nlargest(k_top, metric).index
        intersection = len(set(actual_top).intersection(set(pred_top)))
        enrichment = intersection / k_top
        
        # Random expectation is roughly 20% (since k_top = 20% of N)
        stats[f"enrichment_{metric}"] = float(enrichment)
        
    # Save statistics
    with open(os.path.join(ANALYSIS_DIR, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"[Phase 5] Saved statistics. T_K correlation: {stats['spearman_T_K_acc']['rho']:.3f}")
    
    # 3. Plots
    if plt is not None:
        sns.set_theme(style="whitegrid")
        
        # Scatter: T_K vs Delta_S_K
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='T_K', y='delta_acc', hue='stratum', palette='viridis')
        plt.title(f"T_K vs Intervention Damage (rho={stats['spearman_T_K_acc']['rho']:.3f})")
        plt.xlabel("Capability-Conditioned Sensitivity (T_K)")
        plt.ylabel("Accuracy Drop (ΔS_K)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "scatter_tk_acc.png"))
        plt.close()
        
        # Compare correlations bar plot
        rhos = [
            stats['spearman_A_n_acc']['rho'],
            stats['spearman_W_n_acc']['rho'],
            stats['spearman_T_K_acc']['rho']
        ]
        labels = ['Activation (A_n)', 'Weight (W_n)', 'Sensitivity (T_K)']
        
        plt.figure(figsize=(7, 5))
        sns.barplot(x=labels, y=rhos, palette='muted')
        plt.axhline(0.30, color='red', linestyle='--', label='Go/No-Go Threshold')
        plt.title("Rank Correlation (Spearman ρ) with ΔS_K")
        plt.ylabel("Spearman ρ")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "correlation_comparison.png"))
        plt.close()

        print("[Phase 5] Saved plots.")
    
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_analysis()
