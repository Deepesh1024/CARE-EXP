"""
CARE-MoE Experiment 3A — Phase 2: Random Graph Baselines
======================================================
1. Load CARE capability graphs constructed in Phase 1.
2. Generate 1000 equivalent Erdős-Rényi random graphs (same nodes, same edges).
3. Compute graph topological statistics.
4. Compare CARE graph against random baseline distribution to test H0.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.stats import norm

from config import (
    GRAPHS_DIR,
    BASELINES_DIR,
    LAYERS,
    K_VALUES,
    N_EXPERTS,
    N_RANDOM_GRAPHS
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_pickle,
    save_json
)

def compute_graph_statistics(G: nx.Graph) -> dict:
    """Compute topological statistics for a graph."""
    if G.number_of_edges() == 0:
        return {
            "Density": 0.0,
            "Avg_Degree": 0.0,
            "Clustering_Coef": 0.0,
            "Avg_Path_Length": 0.0,
            "Connected_Components": N_EXPERTS,
            "Modularity": 0.0
        }
        
    stats = {
        "Density": nx.density(G),
        "Avg_Degree": np.mean([d for n, d in G.degree()]),
        "Clustering_Coef": nx.average_clustering(G),
        "Connected_Components": nx.number_connected_components(G)
    }
    
    # Path length (only on largest connected component if disconnected)
    if nx.is_connected(G):
        stats["Avg_Path_Length"] = nx.average_shortest_path_length(G)
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        if subgraph.number_of_nodes() > 1:
            stats["Avg_Path_Length"] = nx.average_shortest_path_length(subgraph)
        else:
            stats["Avg_Path_Length"] = 0.0
            
    # Modularity (using Louvain)
    try:
        partition = community_louvain.best_partition(G, weight=None) # ignore weight for topological comparison
        stats["Modularity"] = community_louvain.modularity(partition, G, weight=None)
    except:
        stats["Modularity"] = 0.0
        
    return stats

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 2 — RANDOM GRAPH BASELINES")
    print("=" * 70)

    results = {}
    significance = {}
    layers = LAYERS + ["aggregated"]

    for layer in layers:
        results[layer] = {}
        significance[layer] = {}
        print(f"\n[Phase 2] Analyzing {layer} layer...")
        
        for k in K_VALUES:
            print(f"  k={k}")
            graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{k}_graph.pkl")
            if not os.path.exists(graph_path):
                print(f"    Warning: Graph not found at {graph_path}")
                continue
                
            G = load_pickle(graph_path)
            
            if G.number_of_edges() == 0:
                print("    Graph has no edges. Skipping random baseline.")
                continue

            # 1. Compute CARE graph statistics
            care_stats = compute_graph_statistics(G)
            
            # 2. Generate Random Graphs
            m = G.number_of_edges()
            random_stats = {key: [] for key in care_stats.keys()}
            
            print(f"    Generating {N_RANDOM_GRAPHS} random graphs (N={N_EXPERTS}, M={m})...")
            for _ in range(N_RANDOM_GRAPHS):
                # Erdős-Rényi graph with exact same number of nodes and edges
                R = nx.gnm_random_graph(N_EXPERTS, m)
                r_stats = compute_graph_statistics(R)
                for key in random_stats.keys():
                    random_stats[key].append(r_stats[key])
            
            # 3. Statistical Comparison
            results[layer][f"k{k}"] = {
                "CARE": care_stats,
                "Random_Mean": {key: float(np.mean(vals)) for key, vals in random_stats.items()},
                "Random_Std": {key: float(np.std(vals)) for key, vals in random_stats.items()}
            }
            
            sig_res = {}
            for key in care_stats.keys():
                care_val = care_stats[key]
                rand_mean = np.mean(random_stats[key])
                rand_std = np.std(random_stats[key])
                
                # Z-score and two-tailed p-value
                if rand_std > 0:
                    z = (care_val - rand_mean) / rand_std
                    p = 2 * (1 - norm.cdf(abs(z)))
                else:
                    z = float('inf') if care_val > rand_mean else (float('-inf') if care_val < rand_mean else 0)
                    p = 0.0 if care_val != rand_mean else 1.0
                
                sig_res[key] = {
                    "CARE_Value": care_val,
                    "Random_Mean": rand_mean,
                    "Z_Score": float(z),
                    "P_Value": float(p),
                    "Significant": bool(p < 0.05)
                }
            
            significance[layer][f"k{k}"] = sig_res
            
            # Print highlights (clustering and modularity)
            print(f"    Modularity: CARE = {care_stats['Modularity']:.4f}, Random = {results[layer][f'k{k}']['Random_Mean']['Modularity']:.4f} (p={sig_res['Modularity']['P_Value']:.4f})")
            print(f"    Clustering: CARE = {care_stats['Clustering_Coef']:.4f}, Random = {results[layer][f'k{k}']['Random_Mean']['Clustering_Coef']:.4f} (p={sig_res['Clustering_Coef']['P_Value']:.4f})")

    # Save results
    save_json(results, os.path.join(BASELINES_DIR, "random_graph_statistics.json"))
    save_json(significance, os.path.join(BASELINES_DIR, "significance_tests.json"))

    print("\n" + "=" * 60)
    print("PHASE 2 — RANDOM BASELINES COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
