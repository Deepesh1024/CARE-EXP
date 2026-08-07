"""
CARE-MoE Experiment 3A — Phase 3: Community Detection
======================================================
1. Apply Louvain to Capability Graphs.
2. Store community assignments.
3. Compute community stability metrics (weighted and unweighted modularity).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain

from config import (
    GRAPHS_DIR,
    COMMUNITIES_DIR,
    LAYERS,
    K_VALUES,
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
    print("PHASE 3 — COMMUNITY DETECTION")
    print("=" * 70)

    layers = LAYERS + ["aggregated"]
    
    assignments_rows = []
    summary_results = {}

    for layer in layers:
        summary_results[layer] = {}
        print(f"\n[Phase 3] Detecting communities in {layer} layer...")
        
        for k in K_VALUES:
            graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{k}_graph.pkl")
            if not os.path.exists(graph_path):
                continue
                
            G = load_pickle(graph_path)
            if G.number_of_edges() == 0:
                continue

            # 1. Louvain (uses edge weights for partitioning)
            louvain_partition = community_louvain.best_partition(G, weight='weight')
            
            # Compute both modularities for the same partition
            weighted_mod = community_louvain.modularity(louvain_partition, G, weight='weight')
            unweighted_mod = community_louvain.modularity(louvain_partition, G, weight=None)
            
            louvain_communities = {}
            for node, comm in louvain_partition.items():
                louvain_communities.setdefault(comm, []).append(node)
                
            n_louvain = len(louvain_communities)
            sizes_louvain = [len(c) for c in louvain_communities.values()]
            
            print(f"    k={k} | Louvain: {n_louvain} communities | W-Mod = {weighted_mod:.4f}, U-Mod = {unweighted_mod:.4f}")
            
            summary_results[layer][f"k{k}"] = {
                "Louvain": {
                    "Num_Communities": n_louvain,
                    "Weighted_Modularity": float(weighted_mod),
                    "Unweighted_Modularity": float(unweighted_mod),
                    "Sizes": sizes_louvain
                }
            }

            # 3. Store assignments
            for exp_id in range(N_EXPERTS):
                row = {
                    "Layer": layer,
                    "k": k,
                    "Expert": exp_id,
                    "Louvain_Community": louvain_partition.get(exp_id, -1)
                }
                assignments_rows.append(row)

    # Save assignments to CSV
    assignments_df = pd.DataFrame(assignments_rows)
    save_csv(assignments_df, os.path.join(COMMUNITIES_DIR, "community_assignments.csv"))
    
    # Save summary to JSON
    save_json(summary_results, os.path.join(COMMUNITIES_DIR, "community_summary.json"))

    print("\n" + "=" * 60)
    print("PHASE 3 — COMMUNITY DETECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
