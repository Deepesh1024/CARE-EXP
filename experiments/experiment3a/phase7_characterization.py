"""
CARE-MoE Experiment 3A — Phase 7: Community Characterization
======================================================
1. Profile each community (size, density, avg KL, degree, hubs).
2. Correlate node centrality with Oracle KL merge sensitivity using
   Pearson, Spearman, and Kendall-tau.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr, kendalltau

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

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 7 — COMMUNITY CHARACTERIZATION")
    print("=" * 70)

    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    if not os.path.exists(comm_path):
        raise FileNotFoundError(f"Missing {comm_path}.")
    assignments = pd.read_csv(comm_path)
    assignments_k = assignments[assignments["k"] == K_PRIMARY]

    pred_path = os.path.join(GRAPHS_DIR, "predictions_df.pkl")
    df = load_pickle(pred_path)

    layers = LAYERS + ["aggregated"]

    for layer in layers:
        print(f"\n[Phase 7] Characterizing {layer} layer (k={K_PRIMARY})...")
        
        layer_assign = assignments_k[assignments_k["Layer"] == layer]
        if layer_assign.empty:
            continue
            
        comm_map = dict(zip(layer_assign["Expert"], layer_assign["Louvain_Community"]))
        
        graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{K_PRIMARY}_graph.pkl")
        G = load_pickle(graph_path)
        
        if layer == "aggregated":
            layer_df = df.groupby(["Expert_A", "Expert_B"], as_index=False)[["Oracle_KL", "Predicted_KL"]].mean()
        else:
            layer_df = df[df["Layer"] == layer]
            
        # Global Centrality
        degree_dict = dict(G.degree())
        betweenness_dict = nx.betweenness_centrality(G)
        
        # Merge Sensitivity = Average Oracle KL when merged with any other expert
        expert_kl_sum = {i: 0.0 for i in range(N_EXPERTS)}
        expert_kl_count = {i: 0 for i in range(N_EXPERTS)}
        
        for _, row in layer_df.iterrows():
            ea, eb = int(row["Expert_A"]), int(row["Expert_B"])
            kl = row["Oracle_KL"]
            expert_kl_sum[ea] += kl
            expert_kl_count[ea] += 1
            expert_kl_sum[eb] += kl
            expert_kl_count[eb] += 1
            
        merge_sensitivity = {i: (expert_kl_sum[i]/expert_kl_count[i] if expert_kl_count[i]>0 else 0) for i in range(N_EXPERTS)}
        
        # 1. Community Profiling
        communities = set(comm_map.values())
        profiles = []
        
        for c in communities:
            if c < 0: continue
            
            nodes = [n for n, comm in comm_map.items() if comm == c]
            subG = G.subgraph(nodes)
            
            # Edges within vs outside
            internal_edges = subG.number_of_edges()
            possible_internal = len(nodes) * (len(nodes) - 1) / 2
            internal_density = internal_edges / possible_internal if possible_internal > 0 else 0
            
            bridge_edges = 0
            for n in nodes:
                for neighbor in G.neighbors(n):
                    if comm_map.get(neighbor) != c:
                        bridge_edges += 1
            bridge_edges = bridge_edges // 2 # undirected
            
            # Find Hubs
            if len(nodes) > 0:
                hub = max(nodes, key=lambda x: degree_dict.get(x, 0))
                
                # Boundary expert: highest number of connections to outside
                boundary = max(nodes, key=lambda x: sum(1 for neighbor in G.neighbors(x) if comm_map.get(neighbor) != c))
            else:
                hub = -1
                boundary = -1
                
            # Avg KLs
            kl_vals = []
            pred_vals = []
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    ea, eb = nodes[i], nodes[j]
                    match = layer_df[((layer_df["Expert_A"] == ea) & (layer_df["Expert_B"] == eb)) | 
                                     ((layer_df["Expert_A"] == eb) & (layer_df["Expert_B"] == ea))]
                    if not match.empty:
                        kl_vals.append(match["Oracle_KL"].values[0])
                        pred_vals.append(match["Predicted_KL"].values[0])
                        
            avg_kl = np.mean(kl_vals) if kl_vals else 0.0
            avg_pred = np.mean(pred_vals) if pred_vals else 0.0
            
            profiles.append({
                "Layer": layer,
                "Community": c,
                "Size": len(nodes),
                "Avg_Oracle_KL": avg_kl,
                "Avg_Predicted_KL": avg_pred,
                "Internal_Density": internal_density,
                "Bridge_Edges": bridge_edges,
                "Avg_Degree": np.mean([degree_dict.get(n, 0) for n in nodes]),
                "Avg_Betweenness": np.mean([betweenness_dict.get(n, 0.0) for n in nodes]),
                "Hub_Expert": hub,
                "Boundary_Expert": boundary
            })
            
        profiles_df = pd.DataFrame(profiles)
        save_csv(profiles_df, os.path.join(VALIDATION_DIR, f"{layer}_community_profiles.csv"))
        
        # 2. Centrality Correlates
        degrees = [degree_dict.get(i, 0) for i in range(N_EXPERTS)]
        betweennesses = [betweenness_dict.get(i, 0.0) for i in range(N_EXPERTS)]
        sensitivities = [merge_sensitivity.get(i, 0.0) for i in range(N_EXPERTS)]
        
        corrs = {}
        for metric_name, x_vals in [("Degree", degrees), ("Betweenness", betweennesses)]:
            pearson_c, pearson_p = pearsonr(x_vals, sensitivities)
            spearman_c, spearman_p = spearmanr(x_vals, sensitivities)
            kendall_c, kendall_p = kendalltau(x_vals, sensitivities)
            
            corrs[metric_name] = {
                "Pearson": {"coef": float(pearson_c), "p": float(pearson_p)},
                "Spearman": {"coef": float(spearman_c), "p": float(spearman_p)},
                "Kendall": {"coef": float(kendall_c), "p": float(kendall_p)}
            }
            
            print(f"    {metric_name} vs Merge Sensitivity:")
            print(f"      Pearson:  {pearson_c:.4f} (p={pearson_p:.4e})")
            print(f"      Spearman: {spearman_c:.4f} (p={spearman_p:.4e})")
            print(f"      Kendall:  {kendall_c:.4f} (p={kendall_p:.4e})")
            
        save_json(corrs, os.path.join(VALIDATION_DIR, f"{layer}_centrality_correlations.json"))

    print("\n" + "=" * 60)
    print("PHASE 7 — CHARACTERIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
