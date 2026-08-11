"""
CARE-MoE Experiment 3B — Phase 1: Distance Matrix Construction
================================================================
1. Load ground-truth Oracle KL from Experiment 1 output.json.
2. Filter to Seq_Len=512 (most reliable calibration size).
3. Construct 64×64 symmetric distance matrices per layer.
4. Run comprehensive sanity checks.
5. Save distance matrices and provenance metadata.

GROUND-TRUTH RULE:
  Only Oracle KL values from output.json are used.
  XGBoost Predicted_KL is NEVER used.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import numpy as np
import pandas as pd

from config import (
    DATA_PATH,
    RESULTS_DIR,
    SEQ_LEN_FILTER,
    N_EXPERTS,
    LAYERS,
    EPSILON,
    RANDOM_SEED,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_raw_data,
    save_csv,
    save_json,
)


def build_distance_matrix(df_layer: pd.DataFrame, n_experts: int) -> np.ndarray:
    """Construct a symmetric distance matrix from Oracle KL values.

    Oracle_KL is inherently symmetric because merging expert i into j
    produces the same merged model as merging j into i (UniformAverage).
    Therefore d(i,j) = Oracle_KL(i,j) is already symmetric.

    Parameters
    ----------
    df_layer : pd.DataFrame
        Filtered DataFrame for one layer, containing Expert_A, Expert_B, Oracle_KL.
        Only upper-triangle pairs (Expert_A < Expert_B).
    n_experts : int
        Number of experts (64).

    Returns
    -------
    np.ndarray
        Symmetric n_experts × n_experts distance matrix with zero diagonal.
    """
    D = np.zeros((n_experts, n_experts))
    for _, row in df_layer.iterrows():
        i, j = int(row["Expert_A"]), int(row["Expert_B"])
        kl_val = float(row["Oracle_KL"])
        D[i, j] = kl_val
        D[j, i] = kl_val  # symmetric
    return D


def sanity_check_distance_matrix(D: np.ndarray, layer_name: str) -> dict:
    """Run comprehensive sanity checks on a distance matrix.

    Returns
    -------
    dict
        Summary of all checks performed.
    """
    n = D.shape[0]
    checks = {"layer": layer_name, "n_experts": n}

    # 1. Diagonal = 0
    diag_values = np.diag(D)
    checks["diagonal_all_zero"] = bool(np.all(diag_values == 0.0))
    checks["diagonal_max"] = float(np.max(np.abs(diag_values)))

    # 2. Symmetry
    symmetry_diff = np.max(np.abs(D - D.T))
    checks["symmetry_max_diff"] = float(symmetry_diff)
    checks["is_symmetric"] = bool(symmetry_diff < EPSILON)

    # 3. Finite values
    checks["all_finite"] = bool(np.all(np.isfinite(D)))
    checks["n_nan"] = int(np.sum(np.isnan(D)))
    checks["n_inf"] = int(np.sum(np.isinf(D)))

    # 4. Non-negative
    checks["all_non_negative"] = bool(np.all(D >= 0))
    checks["min_off_diag"] = float(np.min(D[np.triu_indices(n, k=1)]))

    # 5. Upper-triangle statistics
    upper_tri = D[np.triu_indices(n, k=1)]
    checks["n_pairwise"] = int(len(upper_tri))
    checks["min"] = float(np.min(upper_tri))
    checks["max"] = float(np.max(upper_tri))
    checks["mean"] = float(np.mean(upper_tri))
    checks["median"] = float(np.median(upper_tri))
    checks["std"] = float(np.std(upper_tri))
    checks["p25"] = float(np.percentile(upper_tri, 25))
    checks["p75"] = float(np.percentile(upper_tri, 75))

    # 6. Triangle inequality audit
    n_violations = 0
    n_triples = 0
    max_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                n_triples += 1
                # Check all three orderings
                for (a, b, c) in [(i, j, k), (i, k, j), (j, k, i)]:
                    violation = D[a, b] - (D[a, c] + D[c, b])
                    if violation > EPSILON:
                        n_violations += 1
                        max_violation = max(max_violation, violation)

    checks["n_triples_checked"] = n_triples
    checks["n_triangle_violations"] = n_violations
    checks["max_triangle_violation"] = float(max_violation)
    checks["triangle_inequality_satisfied"] = (n_violations == 0)

    return checks


def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 3B — PHASE 1: DISTANCE MATRIX CONSTRUCTION")
    print("=" * 70)

    # ── 1. Load Data ──────────────────────────────
    raw_df = load_raw_data()
    df = raw_df[raw_df["Seq_Len"] == SEQ_LEN_FILTER].copy()
    print(f"[Phase 1] Filtered to Seq_Len={SEQ_LEN_FILTER}: {len(df):,} rows")

    # Verify expected pair count
    expected_pairs = math.comb(N_EXPERTS, 2)
    for layer in LAYERS:
        layer_df = df[df["Layer"] == layer]
        assert len(layer_df) == expected_pairs, (
            f"Layer {layer}: expected {expected_pairs} pairs, got {len(layer_df)}"
        )
        # Verify pairs are upper-triangle only
        assert (layer_df["Expert_A"] < layer_df["Expert_B"]).all(), (
            f"Layer {layer}: not all pairs satisfy Expert_A < Expert_B"
        )
    print(f"[Phase 1] Verified: {expected_pairs} pairs per layer (upper triangle, A < B)")

    # ── 2. Construct Distance Matrices ────────────
    distance_matrices = {}
    all_checks = {}

    for layer in LAYERS:
        print(f"\n[Phase 1] Processing layer: {layer}")
        layer_df = df[df["Layer"] == layer]

        # Build distance matrix from Oracle KL
        D = build_distance_matrix(layer_df, N_EXPERTS)
        distance_matrices[layer] = D

        # Save distance matrix
        expert_labels = [f"E{i}" for i in range(N_EXPERTS)]
        D_df = pd.DataFrame(D, columns=expert_labels, index=expert_labels)
        save_csv(D_df, os.path.join(RESULTS_DIR, f"oracle_distance_matrix_{layer}.csv"))

        # Sanity checks
        checks = sanity_check_distance_matrix(D, layer)
        all_checks[layer] = checks

        print(f"  Diagonal all zero:     {checks['diagonal_all_zero']}")
        print(f"  Symmetric:             {checks['is_symmetric']}")
        print(f"  All finite:            {checks['all_finite']}")
        print(f"  All non-negative:      {checks['all_non_negative']}")
        print(f"  Min distance:          {checks['min']:.6f}")
        print(f"  Max distance:          {checks['max']:.6f}")
        print(f"  Mean distance:         {checks['mean']:.6f}")
        print(f"  Median distance:       {checks['median']:.6f}")
        print(f"  Triangle violations:   {checks['n_triangle_violations']} / "
              f"{checks['n_triples_checked'] * 3} checks")

    # ── 3. Save Distance Metadata ─────────────────
    distance_metadata = {
        "construction_method": "Oracle_KL used directly as symmetric distance",
        "justification": (
            "Oracle_KL measures KL(P_original || P_merged) where the merged model "
            "is created by UniformAverage of experts i and j. Since merge(i,j) = merge(j,i), "
            "Oracle_KL is inherently symmetric. No symmetrization formula needed."
        ),
        "source_file": DATA_PATH,
        "seq_len_filter": SEQ_LEN_FILTER,
        "n_experts": N_EXPERTS,
        "n_pairwise_per_layer": expected_pairs,
        "layers": LAYERS,
        "raw_distributions_available": False,
        "directional_kl_available": False,
        "symmetric_oracle_kl_available": True,
        "distance_formula": "d(i,j) = Oracle_KL(i,j)  [inherently symmetric]",
        "xgboost_predictions_used": False,
        "sanity_checks": all_checks,
    }
    save_json(distance_metadata, os.path.join(RESULTS_DIR, "distance_metadata.json"))

    # ── 4. Save Data Provenance ───────────────────
    provenance = {
        "experiment": "3B — Capability Geometry Validation Phase A",
        "ground_truth_source": DATA_PATH,
        "ground_truth_field": "Oracle_KL",
        "ground_truth_description": (
            "Per-token mean KL divergence from original model output to merged-expert "
            "model output, computed on calibration data. Measured by Experiment 1 "
            "Oracle benchmark (CARE-Oracle-v1.0)."
        ),
        "seq_len_filter": SEQ_LEN_FILTER,
        "n_experts": N_EXPERTS,
        "layers_analyzed": LAYERS,
        "pairs_per_layer": expected_pairs,
        "total_oracle_measurements_used": expected_pairs * len(LAYERS),
        "surrogate_predictions_excluded": True,
        "exclusion_justification": (
            "XGBoost Predicted_KL from Experiment 2 is an approximation of Oracle KL. "
            "Using it as ground truth would circularly define the latent space whose "
            "existence we are testing."
        ),
        "random_seed": RANDOM_SEED,
    }
    save_json(provenance, os.path.join(RESULTS_DIR, "data_provenance.json"))

    print("\n" + "=" * 70)
    print("PHASE 1 — DISTANCE MATRIX CONSTRUCTION COMPLETE")
    print("=" * 70)

    return distance_matrices


if __name__ == "__main__":
    main()
