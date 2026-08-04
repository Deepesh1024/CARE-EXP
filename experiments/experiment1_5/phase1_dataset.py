"""
CARE-MoE Experiment 1.5 — Phase 1: Dataset Engineering
========================================================
Construct a scientifically valid regression dataset with a Strict Disjoint
Expert Split that guarantees zero information leakage between train and test.

Outputs
-------
    results/exp1_5/train_df.parquet
    results/exp1_5/test_df.parquet
    results/exp1_5/models/scaler.pkl
    stdout: sample counts and discarded cross-boundary pairs
"""

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import (
    FEATURES,
    EXCLUDE_COLS,
    TARGET,
    TRAIN_EXPERTS,
    TEST_EXPERTS,
    SEQ_LEN_FILTER,
    LAYER_DEPTH_MAP,
    TRAIN_PARQUET,
    TEST_PARQUET,
    SCALER_PATH,
    RANDOM_SEED,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_raw_data,
    save_pickle,
)


# ──────────────────────────────────────────────
# Step 1 — Load
# ──────────────────────────────────────────────
def load_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the target Seq_Len slice and validate columns."""
    df_filtered = df[df["Seq_Len"] == SEQ_LEN_FILTER].copy()
    print(f"[Phase 1] Filtered to Seq_Len={SEQ_LEN_FILTER}: {len(df_filtered):,} rows")

    # Sanity-check that all expected feature columns exist
    missing = [f for f in FEATURES if f not in df_filtered.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    if TARGET not in df_filtered.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    return df_filtered


# ──────────────────────────────────────────────
# Step 2–3 — Select Features & Target
# ──────────────────────────────────────────────
def select_features_and_target(df: pd.DataFrame):
    """Return (X DataFrame of features, y Series of target).

    Only numerical feature columns are kept; excluded diagnostic columns
    are dropped.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Add relative layer depth as a column on the dataframe (for later
    # use in Model B and C), but NOT in X yet — that happens in Phase 2.
    df["Relative_Depth"] = df["Layer"].map(LAYER_DEPTH_MAP)

    print(f"[Phase 1] Selected {len(FEATURES)} features, target='{TARGET}'")
    return X, y


# ──────────────────────────────────────────────
# Step 4 — Strict Disjoint Expert Split
# ──────────────────────────────────────────────
def strict_disjoint_split(df: pd.DataFrame):
    """Split rows into train / test with zero cross-boundary leakage.

    Training rows : Expert_A ∈ TRAIN_EXPERTS AND Expert_B ∈ TRAIN_EXPERTS
    Testing rows  : Expert_A ∈ TEST_EXPERTS  AND Expert_B ∈ TEST_EXPERTS
    Discarded     : everything else (cross-boundary pairs)

    Returns
    -------
    train_df, test_df, n_discarded
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

    # ── Leakage validation ──────────────────
    train_experts = set(train_df["Expert_A"]).union(set(train_df["Expert_B"]))
    test_experts = set(test_df["Expert_A"]).union(set(test_df["Expert_B"]))
    overlap = train_experts & test_experts
    if overlap:
        raise RuntimeError(
            f"LEAKAGE DETECTED: experts {overlap} appear in both train and test."
        )

    print(f"[Phase 1] Strict Disjoint Expert Split:")
    print(f"          Training samples  : {len(train_df):>6,}")
    print(f"          Testing samples   : {len(test_df):>6,}")
    print(f"          Discarded (cross) : {n_discarded:>6,}")
    print(f"          Train experts     : {sorted(train_experts)[:5]}...{sorted(train_experts)[-3:]}")
    print(f"          Test experts      : {sorted(test_experts)[:5]}...{sorted(test_experts)[-3:]}")

    return train_df, test_df, n_discarded


# ──────────────────────────────────────────────
# Step 5 — Feature Scaling
# ──────────────────────────────────────────────
def fit_and_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
):
    """Fit RobustScaler on training features; transform both sets.

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled feature matrices)
    y_train, y_test : np.ndarray  (target vectors)
    scaler          : fitted RobustScaler
    """
    scaler = RobustScaler()

    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    # Store scaled features back into DataFrames for persistence
    for i, col in enumerate(feature_cols):
        train_df.loc[:, col] = X_train[:, i]
        test_df.loc[:, col] = X_test[:, i]

    print(f"[Phase 1] RobustScaler fitted on {X_train.shape[0]} training rows, "
          f"{X_train.shape[1]} features")

    return X_train, X_test, y_train, y_test, scaler


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    set_global_seed()
    ensure_dirs()

    # Step 1: Load
    raw_df = load_raw_data()

    # Step 1b: Filter to target Seq_Len
    df = load_and_filter(raw_df)

    # Step 2–3: Select features and target
    select_features_and_target(df)

    # Step 4: Strict Disjoint Expert Split
    train_df, test_df, n_discarded = strict_disjoint_split(df)

    # Step 5: Scale features (fit on train only)
    X_train, X_test, y_train, y_test, scaler = fit_and_transform(
        train_df, test_df, FEATURES
    )

    # Step 6: Save artifacts
    train_df.to_parquet(TRAIN_PARQUET, index=False)
    print(f"[Phase 1] Saved train_df → {TRAIN_PARQUET}")

    test_df.to_parquet(TEST_PARQUET, index=False)
    print(f"[Phase 1] Saved test_df  → {TEST_PARQUET}")

    save_pickle(scaler, SCALER_PATH)

    # ── Summary ─────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — DATASET ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  Training samples  : {len(train_df):,}")
    print(f"  Testing samples   : {len(test_df):,}")
    print(f"  Discarded pairs   : {n_discarded:,}")
    print(f"  Feature columns   : {len(FEATURES)}")
    print(f"  Target            : {TARGET}")
    print(f"  Layers present    : {sorted(train_df['Layer'].unique())}")
    print(f"  Scaler            : RobustScaler")
    print("=" * 60)

    return train_df, test_df, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
