"""
CARE-MoE: Experiment 3A Null B (Primary H2 Null)
===================================================
Tests whether the observed functional community separation
is stronger than the same graph-building pipeline applied to
randomized pairwise distance assignments.

1. Permutes the upper-triangular predicted KL values.
2. Reconstructs Affinity, Mutual-kNN (k=8), Louvain.
3. Computes T = mean(D_between) - mean(D_within) on TRUE Oracle KL.
4. Repeats 1000 times.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from tqdm.auto import tqdm
from scipy.stats import norm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(_THIS_DIR, "..", "..", "results", "exp3a", "graphs")
OUTPUT_JSON = os.path.join(_THIS_DIR, "..", "..", "results", "exp3a_nullB_results.json")
OUTPUT_MD = os.path.join(_THIS_DIR, "..", "..", "results", "exp3a_nullB_report.md")

N_EXPERTS = 64
K_VAL = 8
N_NULLS = 1000

def fast_mutual_knn(affinity_matrix, k):
    """Vectorized mutual k-NN graph construction."""
    # Find top-k for each row
    np.fill_diagonal(affinity_matrix, -np.inf)
    top_k_idx = np.argsort(affinity_matrix, axis=1)[:, -k:]
    
    # Create adjacency
    adj = np.zeros((N_EXPERTS, N_EXPERTS), dtype=bool)
    for i in range(N_EXPERTS):
        adj[i, top_k_idx[i]] = True
        
    # Mutual condition
    mutual_adj = adj & adj.T
    
    G = nx.Graph()
    G.add_nodes_from(range(N_EXPERTS))
    rows, cols = np.where(mutual_adj)
    # Add unique edges
    for r, c in zip(rows, cols):
        if r < c:
            G.add_edge(r, c)
            
    return G

def compute_statistic(G, oracle_matrix):
    """Run Louvain and compute T."""
    if G.number_of_edges() == 0:
        return np.nan, np.nan, np.nan
        
    louvain_partition = community_louvain.best_partition(G)
    
    within_kl = []
    between_kl = []
    
    for i in range(N_EXPERTS):
        for j in range(i+1, N_EXPERTS):
            c_a = louvain_partition.get(i, -1)
            c_b = louvain_partition.get(j, -1)
            
            kl = oracle_matrix[i, j]
            if c_a == c_b:
                within_kl.append(kl)
            else:
                between_kl.append(kl)
                
    if not within_kl or not between_kl:
        return np.nan, np.nan, np.nan
        
    m_within = np.mean(within_kl)
    m_between = np.mean(between_kl)
    return m_between - m_within, m_within, m_between

def main():
    print("=" * 70)
    print("EXPERIMENT 3A NULL B: EDGE PERMUTATION TEST")
    print("=" * 70)

    pred_df = pd.read_pickle(os.path.join(GRAPHS_DIR, "predictions_df.pkl"))
    layers = ["first", "middle", "last", "aggregated"]
    
    results = {}
    
    # Pre-seed for reproducibility
    rng = np.random.default_rng(42)
    
    for layer in layers:
        print(f"\nProcessing layer: {layer}")
        if layer == "aggregated":
            layer_df = pred_df.groupby(["Expert_A", "Expert_B"], as_index=False).mean(numeric_only=True)
        else:
            layer_df = pred_df[pred_df["Layer"] == layer]
            
        # Extract pairs to a fixed order
        pairs = []
        for i in range(N_EXPERTS):
            for j in range(i+1, N_EXPERTS):
                mask = (layer_df["Expert_A"] == i) & (layer_df["Expert_B"] == j)
                if not mask.any():
                    # Handle symmetric case in aggregated
                    mask = (layer_df["Expert_A"] == j) & (layer_df["Expert_B"] == i)
                row = layer_df[mask].iloc[0]
                pairs.append((i, j, row["Predicted_KL"], row["Oracle_KL"]))
                
        # Base arrays
        pred_kl_vals = np.array([p[2] for p in pairs])
        true_kl_vals = np.array([p[3] for p in pairs])
        
        # Real Oracle matrix for fast lookup
        oracle_matrix = np.zeros((N_EXPERTS, N_EXPERTS))
        for p in pairs:
            oracle_matrix[p[0], p[1]] = p[3]
            oracle_matrix[p[1], p[0]] = p[3]
            
        # Compute REAL statistic first
        # 1. Affinity
        median_pred = np.median(pred_kl_vals)
        aff_vals = np.exp(-pred_kl_vals / median_pred)
        aff_mat = np.zeros((N_EXPERTS, N_EXPERTS))
        for idx, (i, j, _, _) in enumerate(pairs):
            aff_mat[i, j] = aff_vals[idx]
            aff_mat[j, i] = aff_vals[idx]
            
        # 2. Graph & Louvain
        G_real = fast_mutual_knn(aff_mat, K_VAL)
        T_real, mw_real, mb_real = compute_statistic(G_real, oracle_matrix)
        
        print(f"  Real T = {T_real:.6f} (Within={mw_real:.6f}, Between={mb_real:.6f})")
        
        # Null iterations
        T_nulls = []
        mw_nulls = []
        mb_nulls = []
        
        for _ in tqdm(range(N_NULLS), desc=f"Null B ({layer})"):
            # Permute the predicted KL assignments
            shuffled_pred = rng.permutation(pred_kl_vals)
            
            # 1. Null Affinity
            median_shuffled = np.median(shuffled_pred)
            aff_vals_null = np.exp(-shuffled_pred / median_shuffled)
            aff_mat_null = np.zeros((N_EXPERTS, N_EXPERTS))
            for idx, (i, j, _, _) in enumerate(pairs):
                aff_mat_null[i, j] = aff_vals_null[idx]
                aff_mat_null[j, i] = aff_vals_null[idx]
                
            # 2. Null Graph & Louvain
            G_null = fast_mutual_knn(aff_mat_null, K_VAL)
            T_n, mw_n, mb_n = compute_statistic(G_null, oracle_matrix)
            
            if not np.isnan(T_n):
                T_nulls.append(T_n)
                mw_nulls.append(mw_n)
                mb_nulls.append(mb_n)
                
        # Stats
        T_nulls = np.array(T_nulls)
        p_val = np.mean(T_nulls >= T_real)
        z_score = (T_real - np.mean(T_nulls)) / (np.std(T_nulls) + 1e-10)
        ci_95 = [float(np.percentile(T_nulls, 2.5)), float(np.percentile(T_nulls, 97.5))]
        
        print(f"  Null T Mean = {np.mean(T_nulls):.6f}, Std = {np.std(T_nulls):.6f}")
        print(f"  Z-Score = {z_score:.2f}, p-value = {p_val:.4f}")
        
        results[layer] = {
            "Real_T": float(T_real),
            "Real_Within_Mean": float(mw_real),
            "Real_Between_Mean": float(mb_real),
            "Null_T_Mean": float(np.mean(T_nulls)),
            "Null_T_Std": float(np.std(T_nulls)),
            "Null_T_95CI": ci_95,
            "Z_Score": float(z_score),
            "p_value": float(p_val)
        }
        
    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate MD report
    lines = [
        "# EXPERIMENT 3A: NULL B AUDIT (EDGE PERMUTATION)",
        "",
        "## Hypothesis H2",
        "> *The assignment of functional divergence values to specific expert pairs contains non-random community structure.*",
        "",
        "## Methodology",
        "We randomized the Predicted KL values across the unordered expert pairs 1000 times.",
        "For each null realization, we reconstructed the Affinity matrix, Mutual-kNN graph ($k=8$), and Louvain communities.",
        "We then computed the exact same statistic $T = D_{between} - D_{within}$ using the *true* Oracle KL.",
        "",
        "## Results",
        ""
    ]
    
    for layer, res in results.items():
        lines.extend([
            f"### Layer: {layer.capitalize()}",
            f"- **Real $T$**: {res['Real_T']:.6f}",
            f"- **Null $T$ (95% CI)**: [{res['Null_T_95CI'][0]:.6f}, {res['Null_T_95CI'][1]:.6f}]",
            f"- **Z-Score**: {res['Z_Score']:.2f}",
            f"- **Empirical p-value**: {res['p_value']:.4f}",
            ""
        ])
        
    lines.extend([
        "## Conclusion",
        "If $p < 0.05$ across layers, Null B is rejected, confirming that the specific pairing of experts contains non-random functional structure beyond just the density distribution of distances."
    ])
    
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
        
    print(f"\nSaved results to {OUTPUT_JSON} and {OUTPUT_MD}")

if __name__ == "__main__":
    main()
