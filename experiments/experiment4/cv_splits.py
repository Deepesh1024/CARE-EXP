"""
CARE-MoE Experiment 4 — Expert-Disjoint CV Splits
===================================================
Generates and freezes all 15 fold assignments (5 partitions × 3 folds).

CRITICAL PROPERTIES:
  - Unit of generalization is the EXPERT, not the pair.
  - Each partition assigns all 64 experts to 3 folds.
  - Each expert appears in the test set exactly ONCE per partition.
  - Fold assignments are deterministic (seeded) and frozen on first run.
  - NEVER regenerated — loaded from cv_splits.json on resume.

Leakage prevention:
  - Train pairs: both experts in train set.
  - Test pairs:  both experts in test set.
  - Cross pairs (one train, one test): DISCARDED.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_EXPERTS,
    N_PARTITIONS,
    N_FOLDS,
    PARTITION_SEEDS,
    CV_SPLITS_PATH,
    RESULTS_DIR,
)


# ══════════════════════════════════════════════════════════
# Split Generation
# ══════════════════════════════════════════════════════════

def generate_partition(seed: int) -> list[dict]:
    """Generate one 3-fold expert-disjoint partition.

    Shuffles all 64 experts deterministically, then splits into 3 folds
    (roughly 21-22 experts each). Each expert appears as test expert
    exactly once across the 3 folds.

    Returns list of 3 dicts with keys 'train_experts', 'test_experts'.
    """
    rng = np.random.RandomState(seed)
    expert_ids = np.arange(N_EXPERTS)
    rng.shuffle(expert_ids)

    # Split into N_FOLDS groups — some folds may have one more expert
    fold_groups = np.array_split(expert_ids, N_FOLDS)

    folds = []
    for fold_idx in range(N_FOLDS):
        test_experts = sorted(fold_groups[fold_idx].tolist())
        train_experts = sorted(
            e for g_idx, g in enumerate(fold_groups)
            if g_idx != fold_idx
            for e in g.tolist()
        )
        folds.append({
            "fold": fold_idx,
            "train_experts": train_experts,
            "test_experts": test_experts,
        })

    return folds


def get_pair_mask(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    expert_set: set,
    mode: str,
) -> np.ndarray:
    """Return boolean mask over pairs.

    mode='train': both pair_i and pair_j in expert_set.
    mode='test':  both pair_i and pair_j in expert_set.
    """
    assert mode in ("train", "test")
    mask_i = np.array([int(x) in expert_set for x in pair_i])
    mask_j = np.array([int(x) in expert_set for x in pair_j])
    return mask_i & mask_j


# ══════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════

def validate_splits(splits: list) -> None:
    """Validate all splits against leakage rules."""
    assert len(splits) == N_PARTITIONS, f"Expected {N_PARTITIONS} partitions"

    for p_idx, partition in enumerate(splits):
        assert len(partition["folds"]) == N_FOLDS, (
            f"Partition {p_idx}: expected {N_FOLDS} folds"
        )

        # Each expert must appear in test exactly once
        all_test_experts = []
        for fold in partition["folds"]:
            test = set(fold["test_experts"])
            train = set(fold["train_experts"])

            # No overlap
            overlap = train & test
            assert not overlap, (
                f"Partition {p_idx} fold {fold['fold']}: "
                f"train/test overlap: {sorted(overlap)}"
            )

            # Train ∪ test = all experts
            union = train | test
            assert union == set(range(N_EXPERTS)), (
                f"Partition {p_idx} fold {fold['fold']}: "
                f"train ∪ test != all experts"
            )

            # Sizes approximately correct
            n_test = len(test)
            assert 20 <= n_test <= 23, (
                f"Partition {p_idx} fold {fold['fold']}: "
                f"unexpected test size {n_test}"
            )

            all_test_experts.extend(fold["test_experts"])

        # Each expert held out exactly once per partition
        assert sorted(all_test_experts) == list(range(N_EXPERTS)), (
            f"Partition {p_idx}: experts not held out exactly once. "
            f"Got: {sorted(all_test_experts)}"
        )

    print(f"[cv_splits] Validation passed: {N_PARTITIONS} partitions × {N_FOLDS} folds")


# ══════════════════════════════════════════════════════════
# Freeze / Load
# ══════════════════════════════════════════════════════════

def freeze_splits(splits: list) -> None:
    """Save splits to disk. Must only be called once."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CV_SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[cv_splits] Splits frozen → {CV_SPLITS_PATH}")


def load_splits() -> list:
    """Load frozen splits from disk. Validates integrity."""
    if not os.path.exists(CV_SPLITS_PATH):
        raise FileNotFoundError(
            f"cv_splits.json not found: {CV_SPLITS_PATH}\n"
            "Run Phase 1 (freeze splits) first."
        )
    with open(CV_SPLITS_PATH, "r") as f:
        splits = json.load(f)

    assert len(splits) == N_PARTITIONS
    for p in splits:
        assert p["partition_seed"] in PARTITION_SEEDS
        assert len(p["folds"]) == N_FOLDS

    print(f"[cv_splits] Loaded frozen splits: {N_PARTITIONS} partitions × {N_FOLDS} folds")
    return splits


def generate_and_freeze_splits() -> list:
    """Generate all splits, validate, and freeze to disk.

    If cv_splits.json already exists, loads and returns existing splits
    (idempotent — safe to call on resume).
    """
    if os.path.exists(CV_SPLITS_PATH):
        print(f"[cv_splits] cv_splits.json already exists — loading frozen splits")
        return load_splits()

    splits = []
    for p_idx, seed in enumerate(PARTITION_SEEDS):
        folds = generate_partition(seed)
        splits.append({
            "partition": p_idx,
            "partition_seed": seed,
            "folds": folds,
        })
        n_test = [len(f["test_experts"]) for f in folds]
        print(f"[cv_splits] Partition {p_idx} (seed={seed}): "
              f"test sizes per fold = {n_test}")

    validate_splits(splits)
    freeze_splits(splits)
    return splits


# ══════════════════════════════════════════════════════════
# Fold Accessor
# ══════════════════════════════════════════════════════════

def get_fold_data(
    splits: list,
    partition_idx: int,
    fold_idx: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Extract train/test arrays for a given partition/fold.

    Returns
    -------
    dict with keys:
        train_experts, test_experts,
        X_train, y_train, pi_train, pj_train,
        X_test, y_test, pi_test, pj_test,
        n_train_pairs, n_test_pairs
    """
    fold = splits[partition_idx]["folds"][fold_idx]
    train_set = set(fold["train_experts"])
    test_set = set(fold["test_experts"])

    train_mask = get_pair_mask(pair_i, pair_j, train_set, "train")
    test_mask = get_pair_mask(pair_i, pair_j, test_set, "test")

    # Sanity: no pair is both train and test
    assert not np.any(train_mask & test_mask), "Pair in both train and test!"

    return {
        "train_experts": fold["train_experts"],
        "test_experts": fold["test_experts"],
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "pi_train": pair_i[train_mask],
        "pj_train": pair_j[train_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "pi_test": pair_i[test_mask],
        "pj_test": pair_j[test_mask],
        "n_train_pairs": int(train_mask.sum()),
        "n_test_pairs": int(test_mask.sum()),
    }


if __name__ == "__main__":
    import numpy as np
    splits = generate_and_freeze_splits()
    # Quick printout
    for p in splits:
        for f in p["folds"]:
            print(f"  P{p['partition']} F{f['fold']}: "
                  f"train={len(f['train_experts'])} test={len(f['test_experts'])} experts")
