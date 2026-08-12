"""
CARE-MoE Experiment 4 — Data Loader
=====================================
Loads and validates:
  1. Oracle distance matrix (64×64) from Experiment 3B middle layer.
  2. Raw pairwise feature data from Experiment 1 (Seq_Len=512, Layer=middle).
  3. Computes all 11 local features (unscaled) aligned to Oracle matrix indices.

Oracle_KL alignment: Exp 3B distance matrix was built directly from
output.json Oracle_KL at Seq_Len=512. Cross-check shows floating-point
differences < 1e-15 — effectively identical.

Feature scaling is NOT done here — must be done per fold (fit on train only).

IMPORTANT: No Oracle KL values enter the feature matrix X.
           Per-expert marginal stats use only pre-merge routing data.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ORACLE_MATRIX_PATH,
    RAW_DATA_PATH,
    N_EXPERTS,
    N_PAIRS,
    LAYER,
    SEQ_LEN,
    LOCAL_FEATURES,
    ORIGINAL_FEATURES,
    FORBIDDEN_FEATURES,
    EPSILON,
)


# ══════════════════════════════════════════════════════════
# Oracle Matrix
# ══════════════════════════════════════════════════════════

def load_oracle_matrix() -> tuple[np.ndarray, str]:
    """Load and validate the Exp 3B middle-layer Oracle distance matrix.

    Returns
    -------
    D : np.ndarray, shape (64, 64), float32
        Symmetric Oracle KL distance matrix.
    oracle_hash : str
        SHA256 of the float64 matrix bytes.
    """
    if not os.path.exists(ORACLE_MATRIX_PATH):
        raise FileNotFoundError(
            f"Oracle matrix not found: {ORACLE_MATRIX_PATH}\n"
            "Ensure Experiment 3B results exist."
        )

    df = pd.read_csv(ORACLE_MATRIX_PATH)
    # All columns are E0..E63 (expert labels)
    D = df.values.astype(np.float64)

    assert D.shape == (N_EXPERTS, N_EXPERTS), (
        f"Oracle matrix shape {D.shape} != ({N_EXPERTS}, {N_EXPERTS})"
    )
    assert np.allclose(D, D.T, atol=1e-10), "Oracle matrix not symmetric"
    assert np.allclose(np.diag(D), 0.0, atol=1e-10), "Oracle diagonal not zero"
    assert not np.any(np.isnan(D)), "Oracle matrix contains NaN"
    assert not np.any(np.isinf(D)), "Oracle matrix contains Inf"

    i_idx, j_idx = np.triu_indices(N_EXPERTS, k=1)
    vals = D[i_idx, j_idx]
    assert len(vals) == N_PAIRS
    assert np.all(vals >= 0), "Negative Oracle values"

    oracle_hash = hashlib.sha256(D.tobytes()).hexdigest()

    print(f"[data_loader] Oracle matrix: {D.shape}, "
          f"range [{vals.min():.6f}, {vals.max():.6f}]")
    print(f"  SHA256: {oracle_hash}")

    return D.astype(np.float32), oracle_hash


# ══════════════════════════════════════════════════════════
# Per-Expert Marginal Statistics (for 3 flagged descriptors)
# ══════════════════════════════════════════════════════════

def _compute_per_expert_stats(df: pd.DataFrame) -> dict:
    """Compute per-expert marginal statistics across ALL 2016 pairs.

    Uses pre-merge routing data only. No Oracle KL content.
    Returns dict: expert_id -> {"Usage_Mean": float}

    NOTE: These stats use all 64 experts' routing data (pre-merge).
    Usage_Asymmetry, Routing_NPMI_Proxy, and Specialization_Diff
    are flagged features because they depend on experts outside pair (i,j).
    See feature_provenance.py for full audit.
    """
    all_experts = sorted(
        set(df["Expert_A"].unique()) | set(df["Expert_B"].unique())
    )
    stats = {}
    for exp_id in all_experts:
        mask = (df["Expert_A"] == exp_id) | (df["Expert_B"] == exp_id)
        pairs = df[mask]
        stats[int(exp_id)] = {
            "Usage_Mean": float(pairs["Usage_Frequency"].mean()),
        }
    return stats


# ══════════════════════════════════════════════════════════
# CARE Descriptor Computation (exact Exp 2 formulas)
# ══════════════════════════════════════════════════════════

def _compute_usage_asymmetry(
    df: pd.DataFrame, expert_stats: dict
) -> np.ndarray:
    """Usage_Asymmetry = |ū_i - ū_j|."""
    vals = np.zeros(len(df), dtype=np.float32)
    for k, (_, row) in enumerate(df.iterrows()):
        ua = expert_stats.get(int(row["Expert_A"]), {}).get("Usage_Mean", 0.0)
        ub = expert_stats.get(int(row["Expert_B"]), {}).get("Usage_Mean", 0.0)
        vals[k] = abs(ua - ub)
    return vals


def _compute_routing_jsd_proxy(df: pd.DataFrame) -> np.ndarray:
    """Routing_JSD_Proxy = (1 - RS) × (1 - JO).  PAIR-LOCAL."""
    rs = np.clip(df["Routing_Similarity"].values, -1.0, 1.0)
    jo = np.clip(df["Jaccard_Overlap"].values, 0.0, 1.0)
    return ((1.0 - rs) * (1.0 - jo)).astype(np.float32)


def _compute_routing_npmi_proxy(
    df: pd.DataFrame, expert_stats: dict
) -> np.ndarray:
    """Routing_NPMI_Proxy — exact Exp 2 formula."""
    global_mean = np.mean([v["Usage_Mean"] for v in expert_stats.values()])

    vals = np.zeros(len(df), dtype=np.float32)
    for k, (_, row) in enumerate(df.iterrows()):
        u_i = expert_stats.get(int(row["Expert_A"]), {}).get("Usage_Mean", global_mean)
        u_j = expert_stats.get(int(row["Expert_B"]), {}).get("Usage_Mean", global_mean)
        p_i = u_i / max(global_mean * 3, EPSILON)
        p_j = u_j / max(global_mean * 3, EPSILON)
        p_ij = max(float(row["Jaccard_Overlap"]) * float(row["Usage_Frequency"]), EPSILON)

        pmi = np.log(p_ij / max(p_i * p_j, EPSILON))
        neg_log_pij = -np.log(max(p_ij, EPSILON))

        vals[k] = float(pmi / neg_log_pij) if neg_log_pij > EPSILON else 0.0

    return np.clip(vals, -1.0, 1.0)


def _compute_specialization_diff(
    df: pd.DataFrame, expert_stats: dict
) -> np.ndarray:
    """Specialization_Diff = |1/(ū_i+ε) - 1/(ū_j+ε)|."""
    spec = {eid: 1.0 / (v["Usage_Mean"] + EPSILON)
            for eid, v in expert_stats.items()}
    vals = np.zeros(len(df), dtype=np.float32)
    for k, (_, row) in enumerate(df.iterrows()):
        sa = spec.get(int(row["Expert_A"]), 0.0)
        sb = spec.get(int(row["Expert_B"]), 0.0)
        vals[k] = abs(sa - sb)
    return vals


# ══════════════════════════════════════════════════════════
# Raw Feature Extraction
# ══════════════════════════════════════════════════════════

def load_raw_features() -> tuple[pd.DataFrame, str]:
    """Load Seq_Len=512, Layer=middle features from Exp 1 output.json.

    Returns UNSCALED features. Scaling must be done per fold.

    Returns
    -------
    feat_df : pd.DataFrame
        2016 rows × (Expert_A, Expert_B, 11 features, Oracle_KL).
        Sorted by (Expert_A ascending, Expert_B ascending).
    feature_hash : str
        SHA256 of the unscaled feature matrix (float64 bytes), pre-sort.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}")

    with open(RAW_DATA_PATH, "r") as f:
        raw = json.load(f)

    df_all = pd.DataFrame(raw["results"])

    # Filter: Layer=middle, Seq_Len=512 (matches Exp 3B Oracle matrix)
    df = df_all[
        (df_all["Layer"] == LAYER) & (df_all["Seq_Len"] == SEQ_LEN)
    ].copy().reset_index(drop=True)

    if len(df) != N_PAIRS:
        raise ValueError(
            f"Expected {N_PAIRS} pairs (Layer={LAYER}, Seq_Len={SEQ_LEN}), "
            f"got {len(df)}."
        )

    # Validate all 64 experts present
    all_experts = set(df["Expert_A"].unique()) | set(df["Expert_B"].unique())
    if all_experts != set(range(N_EXPERTS)):
        missing = set(range(N_EXPERTS)) - all_experts
        raise ValueError(f"STOP: Missing experts in raw data: {sorted(missing)}")

    # Validate original features present — hard stop if missing
    missing_orig = [f for f in ORIGINAL_FEATURES if f not in df.columns]
    if missing_orig:
        raise ValueError(
            f"STOP: Required original feature(s) missing from output.json: {missing_orig}\n"
            "Do NOT substitute replacement features."
        )

    # Ensure no forbidden columns used
    for col in FORBIDDEN_FEATURES:
        if col in LOCAL_FEATURES:
            raise ValueError(f"FORBIDDEN feature '{col}' in LOCAL_FEATURES list!")

    # Enforce canonical pair ordering: Expert_A < Expert_B
    swap_mask = df["Expert_A"] > df["Expert_B"]
    if swap_mask.any():
        df.loc[swap_mask, ["Expert_A", "Expert_B"]] = (
            df.loc[swap_mask, ["Expert_B", "Expert_A"]].values
        )

    df = df.sort_values(["Expert_A", "Expert_B"]).reset_index(drop=True)

    # Compute per-expert marginal stats (ALL 2016 pairs, pre-merge routing only)
    expert_stats = _compute_per_expert_stats(df)

    # Compute 4 new CARE descriptors (exact Exp 2 formulas)
    df["Usage_Asymmetry"] = _compute_usage_asymmetry(df, expert_stats)
    df["Routing_JSD_Proxy"] = _compute_routing_jsd_proxy(df)
    df["Routing_NPMI_Proxy"] = _compute_routing_npmi_proxy(df, expert_stats)
    df["Specialization_Diff"] = _compute_specialization_diff(df, expert_stats)

    # Validate all 11 features present
    missing_features = [f for f in LOCAL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"STOP: Required feature(s) unavailable: {missing_features}. "
            "Do NOT substitute."
        )

    # Validate no NaN/Inf
    feat_vals = df[LOCAL_FEATURES].values.astype(np.float64)
    assert not np.any(np.isnan(feat_vals)), "NaN in local features"
    assert not np.any(np.isinf(feat_vals)), "Inf in local features"

    feature_hash = hashlib.sha256(feat_vals.tobytes()).hexdigest()

    print(f"[data_loader] Features: {len(df)} pairs, {len(LOCAL_FEATURES)} features, "
          f"Seq_Len={SEQ_LEN}")
    print(f"  Feature hash: {feature_hash}")

    keep_cols = ["Expert_A", "Expert_B"] + list(LOCAL_FEATURES) + ["Oracle_KL"]
    return df[keep_cols].copy(), feature_hash


# ══════════════════════════════════════════════════════════
# Aligned Matrix Builder
# ══════════════════════════════════════════════════════════

def build_pair_arrays(
    feat_df: pd.DataFrame,
    D_oracle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned (X, y, pair_i, pair_j) arrays for all 2016 pairs.

    Oracle matrix D_oracle is the authoritative target.
    Raw Oracle_KL from feat_df used only for cross-validation.

    Returns
    -------
    X        : (2016, 11) float32 — unscaled local features
    y        : (2016,) float32   — Oracle KL targets from D_oracle
    pair_i   : (2016,) int32     — first expert index (i < j)
    pair_j   : (2016,) int32     — second expert index
    """
    # Build fast lookup
    lookup = {}
    for _, row in feat_df.iterrows():
        a, b = int(row["Expert_A"]), int(row["Expert_B"])
        key = (min(a, b), max(a, b))
        lookup[key] = row

    X = np.zeros((N_PAIRS, len(LOCAL_FEATURES)), dtype=np.float32)
    y = np.zeros(N_PAIRS, dtype=np.float32)
    pair_i = np.zeros(N_PAIRS, dtype=np.int32)
    pair_j = np.zeros(N_PAIRS, dtype=np.int32)

    max_mismatch = 0.0
    n_mismatches = 0
    k = 0
    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            key = (i, j)
            if key not in lookup:
                raise ValueError(
                    f"STOP: Missing pair ({i},{j}) in feature data. "
                    "Do NOT substitute."
                )
            row = lookup[key]
            X[k] = [float(row[f]) for f in LOCAL_FEATURES]
            y[k] = float(D_oracle[i, j])
            pair_i[k] = i
            pair_j[k] = j

            # Cross-check alignment (informational, not used)
            diff = abs(float(row["Oracle_KL"]) - float(D_oracle[i, j]))
            if diff > 1e-6:
                n_mismatches += 1
                max_mismatch = max(max_mismatch, diff)
            k += 1

    assert k == N_PAIRS

    if n_mismatches > 0:
        print(f"[data_loader] WARNING: {n_mismatches} Oracle_KL mismatches "
              f"(max diff={max_mismatch:.2e}). D_oracle is authoritative.")
    else:
        print(f"[data_loader] Oracle_KL alignment: all {N_PAIRS} pairs match ✓")

    assert not np.any(np.isnan(X)) and not np.any(np.isinf(X))
    assert not np.any(np.isnan(y)) and not np.any(np.isinf(y))

    return X, y, pair_i, pair_j


# ══════════════════════════════════════════════════════════
# Main Entry
# ══════════════════════════════════════════════════════════

def load_all() -> dict:
    """Load and validate all experiment data. Returns validated dict."""
    print("=" * 60)
    print("EXPERIMENT 4 — DATA LOADING AND VALIDATION")
    print("=" * 60)

    D_oracle, oracle_hash = load_oracle_matrix()
    feat_df, feature_hash = load_raw_features()
    X, y, pair_i, pair_j = build_pair_arrays(feat_df, D_oracle)

    print(f"\n[data_loader] Validation complete.")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  y range: [{y.min():.6f}, {y.max():.6f}]")

    return {
        "D_oracle": D_oracle,
        "oracle_hash": oracle_hash,
        "X_unscaled": X,
        "y": y,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "feature_hash": feature_hash,
        "feat_df": feat_df,
    }


if __name__ == "__main__":
    load_all()
