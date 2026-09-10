"""
EXPERIMENT 7A.51 - PHASE 5: REPLICATION ANALYSIS
================================================
Independent replication of capability-conditioned Taylor sensitivity vs Margin Damage.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None

from config import (
    INTERVENTION_DIR, ANALYSIS_DIR, PLOTS_DIR, EXP7A_RESULTS_DIR, RANDOM_SEED,
    ensure_dirs, mark_task, is_task_completed
)

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
    
    return float(base_rho), float(base_pval), float(ci_lower), float(ci_upper)

def run_analysis():
    task_id = "phase5_analysis"
    if is_task_completed(task_id):
        print("[Phase 5] Analysis already completed. Skipping.")
        return

    ensure_dirs()
    
    res_path = os.path.join(INTERVENTION_DIR, "replication_results.csv")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Missing {res_path}. Run phase4 first.")
        
    df = pd.read_csv(res_path)
    
    # Primary Hypothesis Test
    rho, pval, ci_lower, ci_upper = robust_spearman(df['T_K'], df['margin_damage'])
    
    stats = {
        "sample_size": len(df),
        "spearman_rho": rho,
        "p_value": pval,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "random_seed": RANDOM_SEED,
        "experiment_version": "7A.51 Replication"
    }
    
    with open(os.path.join(ANALYSIS_DIR, "statistical_summary.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"[Phase 5] Replication stats saved. rho={rho:.3f}, p={pval:.4f}")
    
    if plt is not None:
        sns.set_theme(style="whitegrid")
        
        # D. Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='T_K', y='margin_damage', hue='stratum', palette='viridis')
        plt.title(f"T_K vs D_margin (rho={rho:.3f}, p={pval:.4f}, N={len(df)})")
        plt.xlabel("Capability-Conditioned Sensitivity (T_K)")
        plt.ylabel("Margin Damage (D_margin)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "scatter_tk_vs_dmargin.png"))
        plt.close()
        
        # E. Distribution Comparison
        exp7a_csv_path = os.path.join(EXP7A_RESULTS_DIR, "intervention", "sampled_neurons.csv")
        if os.path.exists(exp7a_csv_path):
            exp7a_df = pd.read_csv(exp7a_csv_path)
            
            plt.figure(figsize=(8, 6))
            sns.kdeplot(exp7a_df['T_K'], label='7A Original Sample', fill=True)
            sns.kdeplot(df['T_K'], label='7A.51 Replication Sample', fill=True)
            plt.title("T_K Distribution Comparison")
            plt.xlabel("T_K")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "distribution_comparison.png"))
            plt.close()
        
        print("[Phase 5] Saved plots.")
    
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_analysis()
