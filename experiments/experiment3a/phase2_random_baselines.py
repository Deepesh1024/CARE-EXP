"""
CARE-MoE Experiment 3A — Phase 2: Random Graph Baselines
======================================================
1. Load CARE capability graphs constructed in Phase 1.
2. Generate 1000 equivalent Erdős-Rényi random graphs.
3. Compute unweighted statistics (binary topology).
4. Compare CARE graph against random baseline distributions.
5. Save true empirical distributions.
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

def compute_unweighted_statistics(G: nx.Graph) -> dict:
    if G.number_of_edges() == 0:
        return {
            "Density": 0.0,
            "Avg_Degree": 0.0,
            "Clustering_Coef": 0.0,
            "Connected_Components": N_EXPERTS,
            "Global_Efficiency": 0.0,
            "LCC_Size": 1,
            "LCC_Diameter": 0,
            "Transitivity": 0.0,
            "Unweighted_Modularity": 0.0
        }
        
    stats = {
        "Density": nx.density(G),
        "Avg_Degree": np.mean([d for n, d in G.degree()]),
        "Clustering_Coef": nx.average_clustering(G),
        "Connected_Components": nx.number_connected_components(G),
        "Global_Efficiency": nx.global_efficiency(G),
        "Transitivity": nx.transitivity(G)
    }
    
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(components[0])
    stats["LCC_Size"] = lcc.number_of_nodes()
    
    if lcc.number_of_nodes() > 1:
        stats["LCC_Diameter"] = nx.diameter(lcc)
    else:
        stats["LCC_Diameter"] = 0
            
    # Strictly Unweighted Modularity: create binary copy to avoid weight=None API issue
    try:
        G_binary = nx.Graph()
        G_binary.add_nodes_from(G.nodes())
        G_binary.add_edges_from(G.edges())
        partition = community_louvain.best_partition(G_binary)
        stats["Unweighted_Modularity"] = community_louvain.modularity(partition, G_binary)
    except:
        stats["Unweighted_Modularity"] = 0.0
        
    return stats

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 2 — RANDOM GRAPH BASELINES")
    print("=" * 70)

    results = {}
    significance = {}
    empirical_distributions = {}
    
    layers = LAYERS + ["aggregated"]

    for layer in layers:
        results[layer] = {}
        significance[layer] = {}
        empirical_distributions[layer] = {}
        print(f"\n[Phase 2] Analyzing {layer} layer...")
        
        for k in K_VALUES:
            print(f"  k={k}")
            graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{k}_graph.pkl")
            if not os.path.exists(graph_path):
                continue
                
            G = load_pickle(graph_path)
            if G.number_of_edges() == 0:
                continue

            care_stats = compute_unweighted_statistics(G)
            
            m = G.number_of_edges()
            random_stats_lists = {key: [] for key in care_stats.keys()}
            
            print(f"    Generating {N_RANDOM_GRAPHS} random graphs (N={N_EXPERTS}, M={m})...")
            for _ in range(N_RANDOM_GRAPHS):
                R = nx.gnm_random_graph(N_EXPERTS, m)
                r_stats = compute_unweighted_statistics(R)
                for key, val in r_stats.items():
                    random_stats_lists[key].append(val)
            
            # Save empirical distributions
            empirical_distributions[layer][f"k{k}"] = random_stats_lists
            
            results[layer][f"k{k}"] = {
                "CARE": care_stats,
                "Random_Mean": {key: float(np.mean(vals)) for key, vals in random_stats_lists.items()},
                "Random_Std": {key: float(np.std(vals)) for key, vals in random_stats_lists.items()}
            }
            
            sig_res = {}
            for key in care_stats.keys():
                care_val = care_stats[key]
                rand_mean = np.mean(random_stats_lists[key])
                rand_std = np.std(random_stats_lists[key])
                
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
            
            print(f"    Unweighted Modularity: CARE = {care_stats['Unweighted_Modularity']:.4f}, Random = {sig_res['Unweighted_Modularity']['Random_Mean']:.4f} (p={sig_res['Unweighted_Modularity']['P_Value']:.4e})")

    save_json(results, os.path.join(BASELINES_DIR, "random_graph_statistics.json"))
    save_json(significance, os.path.join(BASELINES_DIR, "significance_tests.json"))
    save_json(empirical_distributions, os.path.join(BASELINES_DIR, "empirical_null_distributions.json"))

    print("\n" + "=" * 60)
    print("PHASE 2 — RANDOM BASELINES COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
