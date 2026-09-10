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
    
    # 1. Robust Spearman Correlations
    stats = {}
    
    def robust_spearman(x, y, n_iterations=1000):
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        base_rho, base_pval = spearmanr(x, y)
        
        boot_rhos = []
        for _ in range(n_iterations):
            idx = np.random.choice(n, n, replace=True)
            if np.std(x[idx]) == 0 or np.std(y[idx]) == 0: continue
            r, _ = spearmanr(x[idx], y[idx])
            if not np.isnan(r): boot_rhos.append(r)
            
        ci_lower = np.percentile(boot_rhos, 2.5) if boot_rhos else base_rho
        ci_upper = np.percentile(boot_rhos, 97.5) if boot_rhos else base_rho
        
        perm_rhos = []
        for _ in range(n_iterations):
            y_shuffled = np.random.permutation(y)
            r, _ = spearmanr(x, y_shuffled)
            perm_rhos.append(r)
            
        perm_pval = np.mean(np.abs(perm_rhos) >= np.abs(base_rho))
        
        return {
            "rho": float(base_rho),
            "scipy_p_value": float(base_pval),
            "bootstrap_ci_95": [float(ci_lower), float(ci_upper)],
            "permutation_p_value": float(perm_pval)
        }
    
    for metric in ['T_K', 'A_n', 'W_n']:
        stats[f"spearman_{metric}_acc"] = robust_spearman(df[metric], df['delta_acc'])
        stats[f"spearman_{metric}_loss"] = robust_spearman(df[metric], df['delta_loss'])
        stats[f"spearman_{metric}_logit_margin"] = robust_spearman(df[metric], df['delta_logit_margin'])
        
    # Save statistics
    with open(os.path.join(ANALYSIS_DIR, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"[Phase 5] Saved statistics. T_K vs Loss correlation: {stats['spearman_T_K_loss']['rho']:.3f}")
    
    # 3. Plots
    if plt is not None:
        sns.set_theme(style="whitegrid")
        
        for int_metric, label in [('delta_acc', 'Accuracy Drop'), ('delta_loss', 'Loss Increase'), ('delta_logit_margin', 'Logit Margin Drop')]:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='T_K', y=int_metric, hue='stratum', palette='viridis')
            rho = stats[f"spearman_T_K_{int_metric.replace('delta_', '')}"]['rho']
            pval = stats[f"spearman_T_K_{int_metric.replace('delta_', '')}"]['permutation_p_value']
            plt.title(f"T_K vs {label} (rho={rho:.3f}, p_perm={pval:.3f})")
            plt.xlabel("Capability-Conditioned Sensitivity (T_K)")
            plt.ylabel(f"{label} (Δ)")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"scatter_tk_{int_metric}.png"))
            plt.close()
        
        # Compare correlations bar plot for delta_loss
        rhos = [
            stats['spearman_A_n_loss']['rho'],
            stats['spearman_W_n_loss']['rho'],
            stats['spearman_T_K_loss']['rho']
        ]
        labels = ['Activation (A_n)', 'Weight (W_n)', 'Sensitivity (T_K)']
        
        plt.figure(figsize=(7, 5))
        sns.barplot(x=labels, y=rhos, hue=labels, palette='muted', legend=False)
        plt.axhline(0.30, color='red', linestyle='--', label='Go/No-Go Threshold')
        plt.title("Rank Correlation (Spearman ρ) with ΔLoss")
        plt.ylabel("Spearman ρ")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "correlation_comparison_loss.png"))
        plt.close()

        print("[Phase 5] Saved plots.")
    
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_analysis()
