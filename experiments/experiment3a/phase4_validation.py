"""
CARE-MoE Experiment 3A — Phase 4: Scientific Validation
======================================================
1. Within-community vs Between-community actual Oracle KL.
2. Silhouette Score using the Oracle KL matrix.
3. Community Compressibility Score (Average Within KL / Global KL).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score

from config import (
    GRAPHS_DIR,
    COMMUNITIES_DIR,
    VALIDATION_DIR,
    LAYERS,
    K_PRIMARY,
    N_EXPERTS
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_pickle,
    save_csv,
    save_json
)

def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd

def bootstrap_ci(data, stat_func=np.mean, n_boot=1000, ci=95):
    if len(data) == 0:
        return (0.0, 0.0)
    boot_stats = [stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def build_distance_matrix(layer_df):
    """Builds an NxN symmetric distance matrix using actual Oracle KL."""
    dist_matrix = np.zeros((N_EXPERTS, N_EXPERTS))
    for _, row in layer_df.iterrows():
        ea, eb = int(row["Expert_A"]), int(row["Expert_B"])
        kl = row["Oracle_KL"]
        dist_matrix[ea, eb] = kl
        dist_matrix[eb, ea] = kl
    return dist_matrix

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 4 — SCIENTIFIC VALIDATION")
    print("=" * 70)

    pred_path = os.path.join(GRAPHS_DIR, "predictions_df.pkl")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing {pred_path}. Run Phase 1 first.")
    df = load_pickle(pred_path)
    
    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    if not os.path.exists(comm_path):
        raise FileNotFoundError(f"Missing {comm_path}. Run Phase 3 first.")
    assignments = pd.read_csv(comm_path)

    assignments_k = assignments[assignments["k"] == K_PRIMARY]

    results = []
    stats_summary = {}
    compressibility_rows = []

    layers = LAYERS + ["aggregated"]

    for layer in layers:
        print(f"\n[Phase 4] Validating {layer} layer (k={K_PRIMARY})...")
        
        layer_assign = assignments_k[assignments_k["Layer"] == layer]
        if layer_assign.empty:
            continue
            
        comm_map = dict(zip(layer_assign["Expert"], layer_assign["Louvain_Community"]))
        
        if layer == "aggregated":
            layer_df = df.groupby(["Expert_A", "Expert_B"], as_index=False)["Oracle_KL"].mean()
        else:
            layer_df = df[df["Layer"] == layer]

        within_kl = []
        between_kl = []
        
        # Build Distance Matrix for Silhouette
        dist_matrix = build_distance_matrix(layer_df)
        labels = np.array([comm_map.get(i, -1) for i in range(N_EXPERTS)])
        
        # Valid nodes for silhouette (assigned to a valid community)
        valid_idx = np.where(labels >= 0)[0]
        if len(np.unique(labels[valid_idx])) > 1:
            # Silhouette Score uses precomputed distance
            sil_score = silhouette_score(
                dist_matrix[np.ix_(valid_idx, valid_idx)], 
                labels[valid_idx], 
                metric="precomputed"
            )
        else:
            sil_score = 0.0

        global_mean_kl = layer_df["Oracle_KL"].mean()
        comm_kl_totals = {}
        comm_kl_counts = {}
        
        for _, row in layer_df.iterrows():
            ea, eb = int(row["Expert_A"]), int(row["Expert_B"])
            kl = row["Oracle_KL"]
            
            c_a = comm_map.get(ea, -1)
            c_b = comm_map.get(eb, -2)
            
            if c_a < 0 or c_b < 0:
                continue
                
            is_within = int(c_a == c_b)
            
            if is_within:
                within_kl.append(kl)
                comm_kl_totals[c_a] = comm_kl_totals.get(c_a, 0) + kl
                comm_kl_counts[c_a] = comm_kl_counts.get(c_a, 0) + 1
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

        # Compressibility Scores
        for comm_id, total_kl in comm_kl_totals.items():
            avg_within_kl = total_kl / comm_kl_counts[comm_id]
            comp_score = avg_within_kl / global_mean_kl
            compressibility_rows.append({
                "Layer": layer,
                "Community": comm_id,
                "Avg_Within_KL": avg_within_kl,
                "Global_Avg_KL": global_mean_kl,
                "Compressibility_Score": comp_score
            })

        n_within = len(within_kl)
        n_between = len(between_kl)
        
        if n_within > 0 and n_between > 0:
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
            print(f"    Silhouette Score (KL)  : {sil_score:.4f}")
            
            stats_summary[layer] = {
                "N_Within": n_within,
                "N_Between": n_between,
                "Mean_Within": float(mean_within),
                "Mean_Between": float(mean_between),
                "CI_Within": ci_within,
                "CI_Between": ci_between,
                "MannWhitney_p": float(p_val),
                "Cohens_d": float(d),
                "Significant": bool(p_val < 0.05),
                "Silhouette_Score": float(sil_score),
                "Global_Mean_KL": float(global_mean_kl)
            }

    results_df = pd.DataFrame(results)
    save_csv(results_df, os.path.join(VALIDATION_DIR, "within_vs_between_kl.csv"))
    
    comp_df = pd.DataFrame(compressibility_rows)
    save_csv(comp_df, os.path.join(VALIDATION_DIR, "community_compressibility.csv"))
    
    save_json(stats_summary, os.path.join(VALIDATION_DIR, "validation_statistics.json"))

    print("\n" + "=" * 60)
    print("PHASE 4 — VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
