"""
CARE-MoE Experiment 3A — Phase 3: Community Detection
======================================================
1. Apply Louvain (and Leiden if cdlib available) to Capability Graphs.
2. Store community assignments.
3. Compute community stability metrics (modularity).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain

try:
    from cdlib import algorithms
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False

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
    
    if CDLIB_AVAILABLE:
        print("[Phase 3] cdlib available. Will run both Louvain and Leiden.")
    else:
        print("[Phase 3] cdlib NOT available. Falling back to Louvain only.")

    layers = LAYERS + ["aggregated"]
    
    # We will build a unified CSV with all community assignments
    assignments_rows = []
    summary_results = {}

    for layer in layers:
        summary_results[layer] = {}
        print(f"\n[Phase 3] Detecting communities in {layer} layer...")
        
        for k in K_VALUES:
            print(f"  k={k}")
            graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{k}_graph.pkl")
            if not os.path.exists(graph_path):
                continue
                
            G = load_pickle(graph_path)
            
            if G.number_of_edges() == 0:
                print("    Graph has no edges. Skipping.")
                continue

            # 1. Louvain (uses edge weights)
            louvain_partition = community_louvain.best_partition(G, weight='weight')
            louvain_mod = community_louvain.modularity(louvain_partition, G, weight='weight')
            
            louvain_communities = {}
            for node, comm in louvain_partition.items():
                louvain_communities.setdefault(comm, []).append(node)
                
            n_louvain = len(louvain_communities)
            sizes_louvain = [len(c) for c in louvain_communities.values()]
            
            print(f"    Louvain: {n_louvain} communities, Modularity = {louvain_mod:.4f}")
            
            summary_results[layer][f"k{k}"] = {
                "Louvain": {
                    "Num_Communities": n_louvain,
                    "Modularity": float(louvain_mod),
                    "Sizes": sizes_louvain
                }
            }

            # 2. Leiden (if available)
            if CDLIB_AVAILABLE:
                # cdlib algorithms return a NodeClustering object
                try:
                    leiden_coms = algorithms.leiden(G, weights='weight')
                    leiden_partition = {node: idx for idx, comm in enumerate(leiden_coms.communities) for node in comm}
                    # We can use Louvain's modularity function for evaluation to keep it comparable
                    leiden_mod = community_louvain.modularity(leiden_partition, G, weight='weight')
                    
                    n_leiden = len(leiden_coms.communities)
                    sizes_leiden = [len(c) for c in leiden_coms.communities]
                    
                    print(f"    Leiden:  {n_leiden} communities, Modularity = {leiden_mod:.4f}")
                    
                    summary_results[layer][f"k{k}"]["Leiden"] = {
                        "Num_Communities": n_leiden,
                        "Modularity": float(leiden_mod),
                        "Sizes": sizes_leiden
                    }
                except Exception as e:
                    print(f"    Leiden failed: {e}")
                    leiden_partition = None
            else:
                leiden_partition = None

            # 3. Store assignments
            for exp_id in range(N_EXPERTS):
                row = {
                    "Layer": layer,
                    "k": k,
                    "Expert": exp_id,
                    "Louvain_Community": louvain_partition.get(exp_id, -1)
                }
                if leiden_partition is not None:
                    row["Leiden_Community"] = leiden_partition.get(exp_id, -1)
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
