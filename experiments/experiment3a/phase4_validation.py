"""
CARE-MoE Experiment 3A — Phase 4: Scientific Validation
======================================================
Core scientific validation of Experiment 3A:
Do experts assigned to the same community exhibit significantly 
lower Oracle KL divergence than experts assigned to different communities?
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from config import (
    GRAPHS_DIR,
    COMMUNITIES_DIR,
    VALIDATION_DIR,
    LAYERS,
    K_PRIMARY
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_pickle,
    save_csv,
    save_json
)

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd

def bootstrap_ci(data, stat_func=np.mean, n_boot=1000, ci=95):
    """Compute bootstrap confidence interval for a statistic."""
    if len(data) == 0:
        return (0.0, 0.0)
    boot_stats = [stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 4 — SCIENTIFIC VALIDATION")
    print("=" * 70)

    # 1. Load Data
    pred_path = os.path.join(GRAPHS_DIR, "predictions_df.pkl")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing {pred_path}. Run Phase 1 first.")
    df = load_pickle(pred_path)
    
    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    if not os.path.exists(comm_path):
        raise FileNotFoundError(f"Missing {comm_path}. Run Phase 3 first.")
    assignments = pd.read_csv(comm_path)

    # Focus on the primary graph construction (k=5)
    assignments_k5 = assignments[assignments["k"] == K_PRIMARY]

    results = []
    stats_summary = {}

    layers = LAYERS + ["aggregated"]

    for layer in layers:
        print(f"\n[Phase 4] Validating {layer} layer (k={K_PRIMARY})...")
        
        # Get community mapping for this layer
        layer_assign = assignments_k5[assignments_k5["Layer"] == layer]
        if layer_assign.empty:
            continue
            
        comm_map = dict(zip(layer_assign["Expert"], layer_assign["Louvain_Community"]))
        
        # Determine actual Oracle KL for all pairs
        if layer == "aggregated":
            # For aggregated, we use the average Oracle KL across all layers
            layer_df = df.groupby(["Expert_A", "Expert_B"], as_index=False)["Oracle_KL"].mean()
        else:
            layer_df = df[df["Layer"] == layer]

        within_kl = []
        between_kl = []
        
        for _, row in layer_df.iterrows():
            ea, eb = int(row["Expert_A"]), int(row["Expert_B"])
            kl = row["Oracle_KL"]
            
            c_a = comm_map.get(ea, -1)
            c_b = comm_map.get(eb, -2)
            
            # Skip unassigned (-1)
            if c_a < 0 or c_b < 0:
                continue
                
            is_within = int(c_a == c_b)
            
            if is_within:
                within_kl.append(kl)
            else:
                between_kl.append(kl)
                
            results.append({
                "Layer": layer,
                "Expert_A": ea,
                "Expert_B": eb,
                "Oracle_KL": kl,
                "Is_Within": is_within,
                "Comm_A": c_a,
                "Comm_B": c_b
            })

        # Statistical Testing
        n_within = len(within_kl)
        n_between = len(between_kl)
        
        if n_within > 0 and n_between > 0:
            # Mann-Whitney U test (alternative='less' means we test if within < between)
            stat, p_val = mannwhitneyu(within_kl, between_kl, alternative='less')
            
            mean_within = np.mean(within_kl)
            mean_between = np.mean(between_kl)
            
            d = compute_cohens_d(within_kl, between_kl)
            
            ci_within = bootstrap_ci(within_kl)
            ci_between = bootstrap_ci(between_kl)
            
            print(f"    Within-Community  (n={n_within:<4}): Mean KL = {mean_within:.6f} 95% CI {ci_within}")
            print(f"    Between-Community (n={n_between:<4}): Mean KL = {mean_between:.6f} 95% CI {ci_between}")
            print(f"    Mann-Whitney U p-value : {p_val:.4e} (Significant: {p_val < 0.05})")
            print(f"    Cohen's d effect size  : {d:.4f}")
            
            stats_summary[layer] = {
                "N_Within": n_within,
                "N_Between": n_between,
                "Mean_Within": float(mean_within),
                "Mean_Between": float(mean_between),
                "CI_Within": ci_within,
                "CI_Between": ci_between,
                "MannWhitney_p": float(p_val),
                "Cohens_d": float(d),
                "Significant": bool(p_val < 0.05)
            }
        else:
            print(f"    Skipping stats. n_within={n_within}, n_between={n_between}")

    # Save detailed pair results
    results_df = pd.DataFrame(results)
    save_csv(results_df, os.path.join(VALIDATION_DIR, "within_vs_between_kl.csv"))
    
    # Save statistics
    save_json(stats_summary, os.path.join(VALIDATION_DIR, "validation_statistics.json"))

    print("\n" + "=" * 60)
    print("PHASE 4 — VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
