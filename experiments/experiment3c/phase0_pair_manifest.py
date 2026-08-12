"""
EXPERIMENT 3C — PHASE 0: PAIR MANIFEST GENERATION
===================================================
Generates a deterministic stratified pair manifest using validated
Experiment 3B Oracle distance matrices.

For each layer:
  - Load the 3B Oracle 64×64 distance matrix
  - Extract all 2016 unique pairs from the upper triangle
  - Sort by Oracle KL distance
  - Divide into 4 quartiles (504 pairs each)
  - Deterministically sample 96 pairs from each quartile (seed=42)
  - Total: 384 pairs per layer

The SAME 384 pairs per layer are used at checkpoints 10%, 40%, 70%.
The 100% checkpoint uses ALL 2016 pairs (not this manifest).
"""

import os
import json
import random
import numpy as np
import pandas as pd

from config import (
    N_EXPERTS,
    LAYERS,
    RANDOM_SEED,
    EXP3B_MATRIX_FILES,
    PAIR_MANIFEST_FILE,
    RESULTS_DIR,
    ensure_dirs,
)


def load_3b_matrix(layer):
    """Load a validated 3B Oracle distance matrix."""
    path = EXP3B_MATRIX_FILES[layer]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"3B Oracle matrix not found for layer '{layer}': {path}\n"
            "Experiment 3B results must be present to generate the pair manifest."
        )
    df = pd.read_csv(path, header=0)
    matrix = df.values.astype(np.float64)
    assert matrix.shape == (N_EXPERTS, N_EXPERTS), f"Expected (64,64), got {matrix.shape}"
    return matrix


def stratified_sample(matrix, n_per_quartile=96, seed=42):
    """
    Extract all 2016 unique pairs, sort by Oracle KL,
    split into 4 quartiles, sample n_per_quartile from each.
    Returns list of [i, j] pairs (sorted deterministically).
    """
    pairs_with_dist = []
    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            pairs_with_dist.append((i, j, matrix[i, j]))

    assert len(pairs_with_dist) == 2016, f"Expected 2016 pairs, got {len(pairs_with_dist)}"

    # Sort by Oracle KL distance (ascending = high-affinity first)
    pairs_with_dist.sort(key=lambda x: x[2])

    # Divide into 4 quartiles of 504 each
    quartile_size = len(pairs_with_dist) // 4  # 504
    quartiles = []
    for q in range(4):
        start = q * quartile_size
        end = start + quartile_size if q < 3 else len(pairs_with_dist)
        quartiles.append(pairs_with_dist[start:end])

    # Deterministic sampling
    rng = random.Random(seed)
    sampled = []
    quartile_boundaries = []

    for q_idx, quartile in enumerate(quartiles):
        # Record the KL distance boundaries of each quartile
        q_distances = [p[2] for p in quartile]
        quartile_boundaries.append({
            "quartile": q_idx + 1,
            "label": ["high_affinity", "medium_high", "medium_low", "low_affinity"][q_idx],
            "n_available": len(quartile),
            "n_sampled": n_per_quartile,
            "min_kl": float(min(q_distances)),
            "max_kl": float(max(q_distances)),
            "mean_kl": float(np.mean(q_distances)),
        })

        selected = rng.sample(quartile, n_per_quartile)
        sampled.extend(selected)

    # Sort the final sampled pairs by (i, j) for determinism
    sampled_pairs = sorted([[p[0], p[1]] for p in sampled])

    assert len(sampled_pairs) == 4 * n_per_quartile, (
        f"Expected {4 * n_per_quartile} pairs, got {len(sampled_pairs)}"
    )

    return sampled_pairs, quartile_boundaries


def main():
    ensure_dirs()

    print("=" * 70)
    print("EXPERIMENT 3C — PHASE 0: PAIR MANIFEST GENERATION")
    print("=" * 70)

    if os.path.exists(PAIR_MANIFEST_FILE):
        print(f"[Phase 0] Pair manifest already exists: {PAIR_MANIFEST_FILE}")
        with open(PAIR_MANIFEST_FILE, "r") as f:
            existing = json.load(f)
        # Validate it has the right structure
        if all(layer in existing.get("layers", {}) for layer in LAYERS):
            for layer in LAYERS:
                n = len(existing["layers"][layer]["pairs"])
                print(f"  {layer}: {n} pairs")
            print("[Phase 0] Manifest is valid. Skipping regeneration.")
            return
        print("[Phase 0] Manifest is incomplete. Regenerating...")

    manifest = {
        "manifest_version": "1.0",
        "generation_method": "stratified_quartile_from_3B_oracle",
        "seed": RANDOM_SEED,
        "pairs_per_layer": 384,
        "n_per_quartile": 96,
        "source": "exp3b_oracle_distance_matrices",
        "description": (
            "384 expert pairs per layer, stratified across 4 quartiles of "
            "the validated Experiment 3B Oracle KL distance distribution. "
            "Q1 = high-affinity (lowest KL), Q4 = low-affinity (highest KL). "
            "Same pairs used at checkpoints 10%, 40%, 70%."
        ),
        "layers": {},
    }

    for layer in LAYERS:
        print(f"\n[Phase 0] Processing layer: {layer}")
        matrix = load_3b_matrix(layer)
        print(f"  Loaded 3B matrix: {EXP3B_MATRIX_FILES[layer]}")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  KL range: [{matrix[np.triu_indices(64, k=1)].min():.6f}, "
              f"{matrix[np.triu_indices(64, k=1)].max():.6f}]")

        sampled_pairs, quartile_info = stratified_sample(matrix, n_per_quartile=96, seed=RANDOM_SEED)

        manifest["layers"][layer] = {
            "pairs": sampled_pairs,
            "quartile_boundaries": quartile_info,
            "source_matrix": os.path.basename(EXP3B_MATRIX_FILES[layer]),
        }

        print(f"  Selected {len(sampled_pairs)} pairs:")
        for q in quartile_info:
            print(f"    Q{q['quartile']} ({q['label']}): {q['n_sampled']} pairs, "
                  f"KL range [{q['min_kl']:.6f}, {q['max_kl']:.6f}]")

    # Save manifest
    os.makedirs(os.path.dirname(PAIR_MANIFEST_FILE), exist_ok=True)
    with open(PAIR_MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Phase 0] Manifest saved to: {PAIR_MANIFEST_FILE}")
    print("[Phase 0] PAIR MANIFEST GENERATION COMPLETE.")


if __name__ == "__main__":
    main()
