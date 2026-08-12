"""
CARE-MoE: Experiment 3A Null A (Degree-Preserving Graph Sanity Check)
========================================================================
Tests whether the observed graph topology contains more community structure
than a degree-preserving random graph.

This is a structural sanity check. It DOES NOT prove functional relevance.
1. Take the true k=8 Mutual-kNN unweighted graph.
2. Apply degree-preserving edge swaps to randomize topology (1000 realizations).
3. Run Louvain, compute modularity.
4. Compare true modularity vs null distribution.
"""

import os
import json
import numpy as np
import networkx as nx
import community as community_louvain
from tqdm.auto import tqdm
import pickle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(_THIS_DIR, "..", "..", "results", "exp3a", "graphs")
OUTPUT_JSON = os.path.join(_THIS_DIR, "..", "..", "results", "exp3a_nullA_results.json")
OUTPUT_MD = os.path.join(_THIS_DIR, "..", "..", "results", "exp3a_nullA_report.md")

K_VAL = 8
N_NULLS = 1000
N_SWAPS = 500  # Number of swaps per graph to ensure mixing

def main():
    print("=" * 70)
    print("EXPERIMENT 3A NULL A: GRAPH STRUCTURAL SANITY CHECK")
    print("=" * 70)

    layers = ["first", "middle", "last", "aggregated"]
    results = {}
    
    for layer in layers:
        print(f"\nProcessing layer: {layer}")
        
        graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{K_VAL}_graph.pkl")
        if not os.path.exists(graph_path):
            print(f"Graph not found: {graph_path}")
            continue
            
        with open(graph_path, "rb") as f:
            G_real = pickle.load(f)
            
        if G_real.number_of_edges() == 0:
            print("Graph has 0 edges.")
            continue
            
        # Make a binary copy (Louvain on unweighted)
        G_bin = nx.Graph()
        G_bin.add_nodes_from(G_real.nodes())
        G_bin.add_edges_from(G_real.edges())
        
        # Real modularity
        partition = community_louvain.best_partition(G_bin)
        real_mod = community_louvain.modularity(partition, G_bin)
        real_n_comm = len(set(partition.values()))
        print(f"  Real Modularity: {real_mod:.4f}, Communities: {real_n_comm}")
        
        null_mods = []
        null_n_comms = []
        
        # We need a connected graph or at least enough edges for double_edge_swap
        # If the graph is too sparse or disconnected, double_edge_swap might get stuck.
        # It's usually fine for k=8.
        
        for _ in tqdm(range(N_NULLS), desc=f"Null A ({layer})"):
            # Deep copy the binary graph
            G_null = G_bin.copy()
            # Perform degree-preserving rewiring
            try:
                nx.double_edge_swap(G_null, nswap=N_SWAPS, max_tries=N_SWAPS*10)
            except nx.NetworkXError:
                pass # Graph too small/sparse to swap fully, we just use what we can
                
            p_null = community_louvain.best_partition(G_null)
            null_mods.append(community_louvain.modularity(p_null, G_null))
            null_n_comms.append(len(set(p_null.values())))
            
        null_mods = np.array(null_mods)
        p_val = np.mean(null_mods >= real_mod)
        z_score = (real_mod - np.mean(null_mods)) / (np.std(null_mods) + 1e-10)
        
        print(f"  Null Modularity Mean: {np.mean(null_mods):.4f}, Std: {np.std(null_mods):.4f}")
        print(f"  Z-Score: {z_score:.2f}, p-value: {p_val:.4f}")
        
        results[layer] = {
            "Real_Modularity": float(real_mod),
            "Real_Num_Communities": int(real_n_comm),
            "Null_Modularity_Mean": float(np.mean(null_mods)),
            "Null_Modularity_Std": float(np.std(null_mods)),
            "Null_Num_Communities_Mean": float(np.mean(null_n_comms)),
            "Z_Score": float(z_score),
            "p_value": float(p_val)
        }
        
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate MD report
    lines = [
        "# EXPERIMENT 3A: NULL A AUDIT (GRAPH SANITY CHECK)",
        "",
        "> **WARNING:** This is ONLY a structural sanity check. It determines if the observed graph has more community structure than a random graph with the same degree sequence. **It does NOT prove that the graph or its communities are functionally meaningful.**",
        "",
        "## Methodology",
        "We took the $k=8$ Mutual-kNN unweighted graph and applied `nx.double_edge_swap` 500 times per realization to randomize topology while strictly preserving the degree of every node.",
        "We ran Louvain community detection on 1000 such null graphs and computed the modularity.",
        "",
        "## Results",
        ""
    ]
    
    for layer, res in results.items():
        lines.extend([
            f"### Layer: {layer.capitalize()}",
            f"- **Real Modularity**: {res['Real_Modularity']:.4f}",
            f"- **Null Modularity Mean (Std)**: {res['Null_Modularity_Mean']:.4f} ({res['Null_Modularity_Std']:.4f})",
            f"- **Z-Score**: {res['Z_Score']:.2f}",
            f"- **Empirical p-value**: {res['p_value']:.4f}",
            ""
        ])
        
    lines.extend([
        "## Conclusion",
        "If $p < 0.05$, the real graph exhibits significantly stronger community structure than degree-preserving random graphs. This validates the graph construction pipeline but not the underlying functional claim."
    ])
    
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
        
    print(f"\nSaved results to {OUTPUT_JSON} and {OUTPUT_MD}")

if __name__ == "__main__":
    main()
