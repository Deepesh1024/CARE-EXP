"""
CARE-MoE Experiment 3B — Phase 2: SMACOF MDS + Null Models
=============================================================
1. Run non-metric SMACOF on Oracle distance matrices for all q values.
2. Generate Null A (pairwise-shuffled distance) realizations.
3. Generate Null B (random Euclidean) realizations.
4. Run SMACOF on all null realizations.
5. Save full-matrix stress values for comparison.

IMPORTANT:
  - Non-metric MDS (SMACOF) is used because null models may violate metric axioms.
  - Null A: upper-triangle distances are permuted (NOT row/column relabeling).
  - Null B: 64 random points in R^64, Euclidean distances computed.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from config import (
    RESULTS_DIR,
    N_EXPERTS,
    LAYERS,
    Q_VALUES,
    SMACOF_MAX_ITER,
    SMACOF_N_INIT,
    SMACOF_EPS,
    SMACOF_METRIC,
    N_NULL_REALIZATIONS,
    NULL_B_DIM,
    NULL_B_N_POINTS,
    RANDOM_SEED,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_json,
    save_csv,
    load_json,
)


def load_distance_matrix(layer: str) -> np.ndarray:
    """Load a precomputed Oracle distance matrix for a given layer."""
    path = os.path.join(RESULTS_DIR, f"oracle_distance_matrix_{layer}.csv")
    df = pd.read_csv(path)
    # The CSV was saved with save_csv(index=False), but the DataFrame
    # had index labels E0..E63 that were NOT written. Columns are E0..E63.
    # If the first column is non-numeric (expert labels), drop it.
    if df.iloc[:, 0].dtype == object:
        df = df.iloc[:, 1:]
    return df.values.astype(float)


def run_smacof(D: np.ndarray, q: int, seed: int) -> dict:
    """Run non-metric SMACOF MDS embedding.

    Parameters
    ----------
    D : np.ndarray
        Symmetric distance matrix (n × n).
    q : int
        Target embedding dimensionality.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        embedding : np.ndarray (n × q)
        stress : float (raw stress)
        n_iter : int (SMACOF iterations used)
    """
    mds = MDS(
        n_components=q,
        metric_mds=SMACOF_METRIC,
        metric="precomputed",
        init="random",
        max_iter=SMACOF_MAX_ITER,
        n_init=SMACOF_N_INIT,
        eps=SMACOF_EPS,
        random_state=seed,
        normalized_stress="auto",
        n_jobs=-1,
    )
    Z = mds.fit_transform(D)
    return {
        "embedding": Z,
        "stress": float(mds.stress_),
        "n_iter": mds.n_iter_,
    }


def compute_normalized_stress(D: np.ndarray, Z: np.ndarray) -> float:
    """Compute normalized stress-1 (Kruskal's stress).

    stress_1 = sqrt( sum (d_ij - delta_ij)^2 / sum d_ij^2 )

    where d_ij = ||z_i - z_j||_2  and delta_ij = D[i,j].
    """
    n = D.shape[0]
    idx = np.triu_indices(n, k=1)
    delta = D[idx]

    # Compute embedding distances
    from scipy.spatial.distance import pdist, squareform
    d_embed = pdist(Z, metric="euclidean")

    numerator = np.sum((d_embed - delta) ** 2)
    denominator = np.sum(delta ** 2)
    if denominator < 1e-15:
        return float("inf")
    return float(np.sqrt(numerator / denominator))


def generate_null_a(D: np.ndarray, seed: int) -> np.ndarray:
    """Generate a pairwise-shuffled null distance matrix (Null A).

    Procedure:
    1. Extract all unique upper-triangle distances.
    2. Randomly permute the pairwise distance values.
    3. Reconstruct the symmetric matrix.
    4. Set diagonal to zero.

    This destroys expert-identity ↔ distance relationships while
    approximately preserving the empirical distance distribution.
    """
    rng = np.random.RandomState(seed)
    n = D.shape[0]
    idx = np.triu_indices(n, k=1)

    # Extract upper-triangle values and shuffle
    upper_vals = D[idx].copy()
    rng.shuffle(upper_vals)

    # Reconstruct symmetric matrix
    D_null = np.zeros((n, n))
    D_null[idx] = upper_vals
    D_null = D_null + D_null.T  # symmetrize
    # diagonal is already zero
    return D_null


def generate_null_b(seed: int) -> np.ndarray:
    """Generate a random Euclidean null distance matrix (Null B).

    Generate 64 random points in R^64 and compute their pairwise
    Euclidean distances.
    """
    from scipy.spatial.distance import pdist, squareform
    rng = np.random.RandomState(seed)
    points = rng.randn(NULL_B_N_POINTS, NULL_B_DIM)
    D_null = squareform(pdist(points, metric="euclidean"))
    return D_null


def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 3B — PHASE 2: SMACOF MDS + NULL MODELS")
    print("=" * 70)

    # Store all results
    all_results = {}

    for layer in LAYERS:
        print(f"\n{'─' * 60}")
        print(f"LAYER: {layer}")
        print(f"{'─' * 60}")

        # Load Oracle distance matrix
        D_oracle = load_distance_matrix(layer)
        print(f"[Phase 2] Loaded Oracle distance matrix: {D_oracle.shape}")

        layer_results = {
            "oracle": {},
            "null_a": {},
            "null_b": {},
        }

        # ── Oracle MDS ────────────────────────────────
        print(f"\n[Phase 2] Running Oracle SMACOF for q ∈ {Q_VALUES}...")
        for q in Q_VALUES:
            result = run_smacof(D_oracle, q, seed=RANDOM_SEED)
            norm_stress = compute_normalized_stress(D_oracle, result["embedding"])
            layer_results["oracle"][q] = {
                "stress": result["stress"],
                "normalized_stress": norm_stress,
                "n_iter": result["n_iter"],
                "embedding_shape": list(result["embedding"].shape),
            }
            print(f"  q={q:2d}: stress={result['stress']:.6f}, "
                  f"norm_stress={norm_stress:.6f}, n_iter={result['n_iter']}")

        # ── Null A: Pairwise-Shuffled ─────────────────
        print(f"\n[Phase 2] Generating {N_NULL_REALIZATIONS} Null A (pairwise-shuffled) realizations...")
        for q in Q_VALUES:
            stresses = []
            norm_stresses = []
            for r in range(N_NULL_REALIZATIONS):
                seed_a = RANDOM_SEED * 1000 + r * 100 + q
                D_null = generate_null_a(D_oracle, seed=seed_a)
                result = run_smacof(D_null, q, seed=seed_a)
                ns = compute_normalized_stress(D_null, result["embedding"])
                stresses.append(result["stress"])
                norm_stresses.append(ns)

            layer_results["null_a"][q] = {
                "stresses": stresses,
                "normalized_stresses": norm_stresses,
                "mean_stress": float(np.mean(stresses)),
                "std_stress": float(np.std(stresses)),
                "mean_norm_stress": float(np.mean(norm_stresses)),
                "std_norm_stress": float(np.std(norm_stresses)),
            }
            print(f"  q={q:2d}: mean_norm_stress={np.mean(norm_stresses):.6f} ± {np.std(norm_stresses):.6f}")

        # ── Null B: Random Euclidean ──────────────────
        print(f"\n[Phase 2] Generating {N_NULL_REALIZATIONS} Null B (random Euclidean) realizations...")
        for q in Q_VALUES:
            stresses = []
            norm_stresses = []
            for r in range(N_NULL_REALIZATIONS):
                seed_b = RANDOM_SEED * 2000 + r * 100 + q
                D_null = generate_null_b(seed=seed_b)
                result = run_smacof(D_null, q, seed=seed_b)
                ns = compute_normalized_stress(D_null, result["embedding"])
                stresses.append(result["stress"])
                norm_stresses.append(ns)

            layer_results["null_b"][q] = {
                "stresses": stresses,
                "normalized_stresses": norm_stresses,
                "mean_stress": float(np.mean(stresses)),
                "std_stress": float(np.std(stresses)),
                "mean_norm_stress": float(np.mean(norm_stresses)),
                "std_norm_stress": float(np.std(norm_stresses)),
            }
            print(f"  q={q:2d}: mean_norm_stress={np.mean(norm_stresses):.6f} ± {np.std(norm_stresses):.6f}")

        all_results[layer] = layer_results

    # ── Save Full-Matrix Stress Results ──────────────
    save_json(all_results, os.path.join(RESULTS_DIR, "phase2_mds_results.json"))

    # ── Summary Table ────────────────────────────────
    rows = []
    for layer in LAYERS:
        for q in Q_VALUES:
            oracle_ns = all_results[layer]["oracle"][q]["normalized_stress"]
            null_a_ns = all_results[layer]["null_a"][q]["mean_norm_stress"]
            null_b_ns = all_results[layer]["null_b"][q]["mean_norm_stress"]
            rows.append({
                "layer": layer,
                "q": q,
                "oracle_norm_stress": oracle_ns,
                "null_a_mean_norm_stress": null_a_ns,
                "null_a_std_norm_stress": all_results[layer]["null_a"][q]["std_norm_stress"],
                "null_b_mean_norm_stress": null_b_ns,
                "null_b_std_norm_stress": all_results[layer]["null_b"][q]["std_norm_stress"],
            })
    summary_df = pd.DataFrame(rows)
    save_csv(summary_df, os.path.join(RESULTS_DIR, "phase2_stress_summary.csv"))

    print("\n" + "=" * 70)
    print("PHASE 2 — SMACOF MDS + NULL MODELS COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
