"""
CARE-MoE Experiment 2 — Phase 1: Capability Descriptor Engineering
====================================================================
Engineer 4 new pairwise capability descriptors from the raw output.json
data. Each descriptor satisfies all CARE constraints:
  ✓ Computable before expert merging
  ✓ Requires NO merged model
  ✓ Requires NO oracle
  ✓ Requires NO second forward pass
  ✓ Lightweight, explainable, deployable

Descriptors
-----------
1. Usage_Asymmetry     — |usage_i - usage_j| (per-expert marginal usage)
2. Routing_JSD_Proxy   — JSD approximation from routing statistics
3. Routing_NPMI_Proxy  — NPMI approximation from co-activation statistics
4. Specialization_Diff — |specialization_i - specialization_j|

Pipeline
--------
1. Load raw output.json, filter to Seq_Len=256
2. Extract per-expert marginal statistics across all pairs per layer
3. Compute 4 descriptors at pair level
4. Apply identical disjoint expert split (experts 0-31 train, 32-63 test)
5. Merge new descriptors with existing 7 features
6. Apply RobustScaler (fit on train only)
7. Save augmented train/test parquets

Produces:
  results/exp2/train_df.parquet
  results/exp2/test_df.parquet
  results/exp2/models/scaler.pkl
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import (
    ORIGINAL_FEATURES,
    NEW_DESCRIPTORS,
    ALL_FEATURES,
    TARGET,
    TRAIN_EXPERTS,
    TEST_EXPERTS,
    SEQ_LEN_FILTER,
    LAYER_DEPTH_MAP,
    TRAIN_PARQUET,
    TEST_PARQUET,
    SCALER_PATH,
    RANDOM_SEED,
    EPSILON,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_raw_data,
    save_pickle,
)


# ══════════════════════════════════════════════════════════
# Per-Expert Marginal Statistics
# ══════════════════════════════════════════════════════════

def compute_per_expert_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-expert marginal statistics from pairwise data.

    For each expert in each layer, computes statistics by aggregating
    across all pairs containing that expert.

    Strategy
    --------
    Each expert appears in C(63,1)=63 pairs per layer (as Expert_A or
    Expert_B). The mean of pairwise statistics across all partners gives
    an unbiased estimate of the expert's marginal behavior.

    For Usage_Frequency specifically:
      Usage_Frequency = |A∪B| / N = (|A| + |B| - |A∩B|) / N
    The mean of Usage_Frequency(i, *) across all partners converges to
    (|i|/N + E[|j|/N] - E[|i∩j|/N]). The deviation from the global
    mean isolates expert i's individual contribution.

    Returns
    -------
    pd.DataFrame with columns: Layer, Expert, Usage_Mean, Usage_Std,
        WDist_Mean, OutSim_Mean, RoutSim_Mean
    """
    rows = []

    for layer in sorted(df["Layer"].unique()):
        layer_df = df[df["Layer"] == layer]

        # Get all unique experts in this layer
        all_experts = sorted(
            set(layer_df["Expert_A"].unique()) |
            set(layer_df["Expert_B"].unique())
        )

        for exp_id in all_experts:
            # All pairs containing this expert
            mask = (layer_df["Expert_A"] == exp_id) | (layer_df["Expert_B"] == exp_id)
            pairs = layer_df[mask]

            if len(pairs) == 0:
                continue

            rows.append({
                "Layer": layer,
                "Expert": exp_id,
                "Usage_Mean": pairs["Usage_Frequency"].mean(),
                "Usage_Std": pairs["Usage_Frequency"].std(),
                "WDist_Mean": pairs["Weight_Distance"].mean(),
                "OutSim_Mean": pairs["Output_Similarity"].mean(),
                "RoutSim_Mean": pairs["Routing_Similarity"].mean(),
                "JaccOverlap_Mean": pairs["Jaccard_Overlap"].mean(),
                "N_Pairs": len(pairs),
            })

    expert_df = pd.DataFrame(rows)
    print(f"[Phase 1] Computed per-expert stats: {len(expert_df)} experts "
          f"across {expert_df['Layer'].nunique()} layers")

    return expert_df


# ══════════════════════════════════════════════════════════
# Descriptor 1: Usage Asymmetry
# ══════════════════════════════════════════════════════════

def compute_usage_asymmetry(df: pd.DataFrame,
                             expert_stats: pd.DataFrame) -> np.ndarray:
    """Compute Usage_Asymmetry = |usage_i - usage_j|.

    Scientific Motivation
    --------------------
    Existing features treat expert pairs symmetrically. But merging a
    heavily-routed expert (high usage, processes many tokens) with a
    lightly-routed expert (low usage) causes asymmetric capability
    destruction: the heavily-used expert's processing patterns get diluted
    by the lightly-used partner's weights.

    Hypothesis
    ----------
    Pairs with high usage asymmetry exhibit higher Oracle KL divergence
    because the dominant expert's capability gets disproportionately degraded.

    Mathematical Equation
    --------------------
    Δ_usage(i,j) = |ū_i - ū_j|

    where ū_i = mean(Usage_Frequency) across all pairs containing expert i,
    estimated from the pairwise data structure.

    Computational Complexity: O(E) per layer where E = number of experts.
    Memory Complexity: O(E) for the per-expert lookup table.
    Numerical Stability: Always non-negative; bounded by [0, 1].
    """
    # Create lookup: (layer, expert) -> Usage_Mean
    usage_lookup = {}
    for _, row in expert_stats.iterrows():
        usage_lookup[(row["Layer"], row["Expert"])] = row["Usage_Mean"]

    values = np.zeros(len(df))
    for idx, row in df.iterrows():
        usage_a = usage_lookup.get((row["Layer"], row["Expert_A"]), 0.0)
        usage_b = usage_lookup.get((row["Layer"], row["Expert_B"]), 0.0)
        values[df.index.get_loc(idx)] = abs(usage_a - usage_b)

    return values


# ══════════════════════════════════════════════════════════
# Descriptor 2: Routing JSD Proxy
# ══════════════════════════════════════════════════════════

def compute_routing_jsd_proxy(df: pd.DataFrame) -> np.ndarray:
    """Compute Routing_JSD_Proxy from routing statistics.

    Scientific Motivation
    --------------------
    Routing_Similarity uses Pearson correlation, which captures LINEAR
    alignment between routing probability vectors. JSD captures full
    distributional divergence including tail behavior. XGBoost's top-3
    gain features include both Routing_Similarity and Jaccard_Overlap,
    suggesting their INTERACTION carries nonlinear predictive signal
    that linear models cannot access.

    Hypothesis
    ----------
    A multiplicative proxy for JSD, combining routing correlation with
    co-activation overlap, captures the nonlinear distributional divergence
    that contributes to the Linearization Gap.

    Mathematical Equation
    --------------------
    JSD_proxy(i,j) = (1 - Routing_Similarity(i,j)) × (1 - Jaccard_Overlap(i,j))

    When Routing_Similarity is low (distributions point in different
    directions) AND Jaccard_Overlap is low (experts process different
    token sets), the routing distributions are maximally divergent.

    This is the geometric mean of the two routing divergence signals,
    capturing an interaction that linear models cannot represent.

    Computational Complexity: O(1) per pair — trivial.
    Memory Complexity: O(1).
    Numerical Stability: Bounded [0, 4] since both inputs ∈ [-1, 1].
                         Practically bounded [0, 2] for reasonable data.
    """
    # Clamp Routing_Similarity to [-1, 1] for safety
    rs = np.clip(df["Routing_Similarity"].values, -1.0, 1.0)
    jo = np.clip(df["Jaccard_Overlap"].values, 0.0, 1.0)

    return (1.0 - rs) * (1.0 - jo)


# ══════════════════════════════════════════════════════════
# Descriptor 3: Routing NPMI Proxy
# ══════════════════════════════════════════════════════════

def compute_routing_npmi_proxy(df: pd.DataFrame,
                                expert_stats: pd.DataFrame) -> np.ndarray:
    """Compute Routing_NPMI_Proxy from co-activation statistics.

    Scientific Motivation
    --------------------
    NPMI (Normalized Pointwise Mutual Information) measures whether two
    experts co-activate on the same tokens MORE or LESS than expected by
    chance. Experts that systematically co-activate are functionally
    complementary — merging them destroys complementary processing pathways.

    Hypothesis
    ----------
    Expert pairs with high NPMI (co-activated beyond chance) suffer higher
    merge damage because their functions are complementary, not redundant.

    Mathematical Equation
    --------------------
    NPMI(i,j) = log(P(i,j) / (P(i)·P(j))) / (-log(P(i,j)))

    Proxy formulation:
      P(i) ≈ Usage_Mean_i / global_mean (normalized individual usage)
      P(j) ≈ Usage_Mean_j / global_mean
      P(i,j) ≈ Jaccard_Overlap × Usage_Frequency (co-occurrence fraction)

    NPMI ∈ [-1, 1]:
      +1 → perfect co-occurrence (always active together)
       0 → statistical independence
      -1 → mutual exclusion (never active together)

    Computational Complexity: O(1) per pair.
    Memory Complexity: O(E) for per-expert usage lookup.
    Numerical Stability: Uses EPSILON floor to prevent log(0).
    """
    # Per-expert usage lookup
    usage_lookup = {}
    for _, row in expert_stats.iterrows():
        usage_lookup[(row["Layer"], row["Expert"])] = row["Usage_Mean"]

    # Global mean usage per layer
    layer_means = expert_stats.groupby("Layer")["Usage_Mean"].mean().to_dict()

    values = np.zeros(len(df))
    for idx, row in df.iterrows():
        loc = df.index.get_loc(idx)
        layer = row["Layer"]
        global_mean = layer_means.get(layer, 1.0)

        # Marginal probabilities (normalized to [0, 1] range)
        p_i = usage_lookup.get((layer, row["Expert_A"]), global_mean) / max(global_mean * 3, EPSILON)
        p_j = usage_lookup.get((layer, row["Expert_B"]), global_mean) / max(global_mean * 3, EPSILON)

        # Joint probability: co-occurrence fraction
        # Jaccard_Overlap × Usage_Frequency gives |A∩B|/N approximately
        p_ij = max(row["Jaccard_Overlap"] * row["Usage_Frequency"], EPSILON)

        # NPMI computation
        pmi = np.log(p_ij / max(p_i * p_j, EPSILON))
        neg_log_pij = -np.log(max(p_ij, EPSILON))

        if neg_log_pij > EPSILON:
            values[loc] = pmi / neg_log_pij
        else:
            values[loc] = 0.0

    # Clip to theoretical bounds
    return np.clip(values, -1.0, 1.0)


# ══════════════════════════════════════════════════════════
# Descriptor 4: Specialization Difference
# ══════════════════════════════════════════════════════════

def compute_specialization_diff(df: pd.DataFrame,
                                 expert_stats: pd.DataFrame) -> np.ndarray:
    """Compute Specialization_Diff = |spec_i - spec_j|.

    Scientific Motivation
    --------------------
    Experts can be characterized on a generalist-specialist spectrum.
    A generalist (high usage, similar to many others) processes diverse
    tokens; a specialist (low usage, dissimilar to others) handles
    specific token types. Merging a generalist with a specialist dilutes
    the specialist's focused capability.

    Hypothesis
    ----------
    Pairs with high specialization difference suffer higher Oracle KL
    because the specialist's niche capability gets destroyed.

    Mathematical Equation
    --------------------
    specialization_i = 1 / (Usage_Mean_i + ε)

    This is inversely proportional to usage: rarely-used experts are
    specialists (high specialization score), frequently-used experts
    are generalists (low specialization score).

    Δ_spec(i,j) = |specialization_i - specialization_j|
                = |1/(ū_i + ε) - 1/(ū_j + ε)|

    Computational Complexity: O(1) per pair.
    Memory Complexity: O(E) for per-expert lookup.
    Numerical Stability: EPSILON prevents division by zero.
                         Bounded by [0, 1/ε] but practically bounded
                         by the usage frequency range.
    """
    # Compute per-expert specialization score
    spec_lookup = {}
    for _, row in expert_stats.iterrows():
        spec_score = 1.0 / (row["Usage_Mean"] + EPSILON)
        spec_lookup[(row["Layer"], row["Expert"])] = spec_score

    values = np.zeros(len(df))
    for idx, row in df.iterrows():
        loc = df.index.get_loc(idx)
        spec_a = spec_lookup.get((row["Layer"], row["Expert_A"]), 0.0)
        spec_b = spec_lookup.get((row["Layer"], row["Expert_B"]), 0.0)
        values[loc] = abs(spec_a - spec_b)

    return values


# ══════════════════════════════════════════════════════════
# Dataset Construction Pipeline
# ══════════════════════════════════════════════════════════

def build_augmented_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full augmented dataset with all 11 features.

    Steps
    -----
    1. Filter to Seq_Len=256
    2. Compute per-expert marginal statistics
    3. Compute 4 new descriptors
    4. Merge with existing features

    Returns
    -------
    pd.DataFrame : Full Seq_Len=256 slice with 11 features + metadata + target.
    """
    # Step 1: Filter
    df = raw_df[raw_df["Seq_Len"] == SEQ_LEN_FILTER].copy()
    print(f"[Phase 1] Filtered to Seq_Len={SEQ_LEN_FILTER}: {len(df):,} rows")

    # Step 2: Per-expert statistics
    expert_stats = compute_per_expert_stats(df)

    # Step 3: Compute descriptors
    print("[Phase 1] Computing new capability descriptors...")

    df["Usage_Asymmetry"] = compute_usage_asymmetry(df, expert_stats)
    print(f"  Usage_Asymmetry: mean={df['Usage_Asymmetry'].mean():.6f}, "
          f"std={df['Usage_Asymmetry'].std():.6f}")

    df["Routing_JSD_Proxy"] = compute_routing_jsd_proxy(df)
    print(f"  Routing_JSD_Proxy: mean={df['Routing_JSD_Proxy'].mean():.6f}, "
          f"std={df['Routing_JSD_Proxy'].std():.6f}")

    df["Routing_NPMI_Proxy"] = compute_routing_npmi_proxy(df, expert_stats)
    print(f"  Routing_NPMI_Proxy: mean={df['Routing_NPMI_Proxy'].mean():.6f}, "
          f"std={df['Routing_NPMI_Proxy'].std():.6f}")

    df["Specialization_Diff"] = compute_specialization_diff(df, expert_stats)
    print(f"  Specialization_Diff: mean={df['Specialization_Diff'].mean():.6f}, "
          f"std={df['Specialization_Diff'].std():.6f}")

    # Add Relative_Depth
    df["Relative_Depth"] = df["Layer"].map(LAYER_DEPTH_MAP)

    # Validate all features exist
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after descriptor engineering: {missing}")

    return df, expert_stats


def strict_disjoint_split(df: pd.DataFrame):
    """Apply identical expert split as Experiment 1.5.

    Returns train_df, test_df, n_discarded.
    """
    is_train = (
        df["Expert_A"].isin(TRAIN_EXPERTS) & df["Expert_B"].isin(TRAIN_EXPERTS)
    )
    is_test = (
        df["Expert_A"].isin(TEST_EXPERTS) & df["Expert_B"].isin(TEST_EXPERTS)
    )

    train_df = df[is_train].copy()
    test_df = df[is_test].copy()
    n_discarded = len(df) - len(train_df) - len(test_df)

    # Leakage validation
    train_experts = set(train_df["Expert_A"]).union(set(train_df["Expert_B"]))
    test_experts = set(test_df["Expert_A"]).union(set(test_df["Expert_B"]))
    overlap = train_experts & test_experts
    if overlap:
        raise RuntimeError(f"LEAKAGE DETECTED: experts {overlap} in both sets.")

    print(f"[Phase 1] Strict Disjoint Expert Split:")
    print(f"          Training samples  : {len(train_df):>6,}")
    print(f"          Testing samples   : {len(test_df):>6,}")
    print(f"          Discarded (cross) : {n_discarded:>6,}")

    return train_df, test_df, n_discarded


def fit_and_transform(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_cols: list):
    """Fit RobustScaler on training features; transform both sets.

    Returns scaler (fitted).
    """
    scaler = RobustScaler()

    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    # Store scaled values back
    for i, col in enumerate(feature_cols):
        train_df.loc[:, col] = X_train[:, i]
        test_df.loc[:, col] = X_test[:, i]

    print(f"[Phase 1] RobustScaler fitted on {X_train.shape[0]} training rows, "
          f"{X_train.shape[1]} features")

    return scaler


def main():
    set_global_seed()
    ensure_dirs()

    print("=" * 70)
    print("PHASE 1 — CAPABILITY DESCRIPTOR ENGINEERING")
    print("=" * 70)

    # Load raw data
    raw_df = load_raw_data()

    # Build augmented dataset
    df, expert_stats = build_augmented_dataset(raw_df)

    # Split
    train_df, test_df, n_discarded = strict_disjoint_split(df)

    # Scale ALL features (original + new)
    scaler = fit_and_transform(train_df, test_df, ALL_FEATURES)

    # Save
    train_df.to_parquet(TRAIN_PARQUET, index=False)
    print(f"[Phase 1] Saved train_df → {TRAIN_PARQUET}")

    test_df.to_parquet(TEST_PARQUET, index=False)
    print(f"[Phase 1] Saved test_df  → {TEST_PARQUET}")

    save_pickle(scaler, SCALER_PATH)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 — DESCRIPTOR ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  Training samples  : {len(train_df):,}")
    print(f"  Testing samples   : {len(test_df):,}")
    print(f"  Discarded pairs   : {n_discarded:,}")
    print(f"  Original features : {len(ORIGINAL_FEATURES)}")
    print(f"  New descriptors   : {len(NEW_DESCRIPTORS)}")
    print(f"  Total features    : {len(ALL_FEATURES)}")
    print(f"  Layers present    : {sorted(train_df['Layer'].unique())}")
    print("=" * 60)

    return train_df, test_df


if __name__ == "__main__":
    main()
