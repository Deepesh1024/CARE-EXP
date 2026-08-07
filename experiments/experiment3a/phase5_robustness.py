"""
CARE-MoE Experiment 3A — Phase 5: Robustness Analysis
======================================================
Check the stability of the communities discovered across k=3, 5, 8.
Metrics: Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
"""

import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import (
    COMMUNITIES_DIR,
    VALIDATION_DIR,
    LAYERS,
    K_VALUES
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_json
)

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 5 — ROBUSTNESS ANALYSIS")
    print("=" * 70)

    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    if not os.path.exists(comm_path):
        raise FileNotFoundError(f"Missing {comm_path}. Run Phase 3 first.")
    assignments = pd.read_csv(comm_path)

    robustness = {}
    layers = LAYERS + ["aggregated"]

    for layer in layers:
        print(f"\n[Phase 5] Analyzing {layer} layer...")
        layer_assign = assignments[assignments["Layer"] == layer]
        
        # We need community vectors for k=3, 5, 8
        vectors = {}
        for k in K_VALUES:
            k_df = layer_assign[layer_assign["k"] == k].sort_values("Expert")
            if not k_df.empty:
                vectors[k] = k_df["Louvain_Community"].values
                
        if len(vectors) < 2:
            print("  Not enough k values to compare.")
            continue
            
        layer_robustness = {}
        
        # Pairwise comparisons
        pairs = [(3, 5), (5, 8), (3, 8)]
        for k1, k2 in pairs:
            if k1 in vectors and k2 in vectors:
                v1, v2 = vectors[k1], vectors[k2]
                ari = adjusted_rand_score(v1, v2)
                nmi = normalized_mutual_info_score(v1, v2)
                
                layer_robustness[f"k{k1}_vs_k{k2}"] = {
                    "ARI": float(ari),
                    "NMI": float(nmi)
                }
                print(f"  k={k1} vs k={k2} -> ARI: {ari:.4f}, NMI: {nmi:.4f}")
                
        robustness[layer] = layer_robustness

    save_json(robustness, os.path.join(VALIDATION_DIR, "robustness_ari_nmi.json"))

    print("\n" + "=" * 60)
    print("PHASE 5 — ROBUSTNESS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
