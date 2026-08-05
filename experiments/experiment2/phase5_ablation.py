"""
CARE-MoE Experiment 2 — Phase 5: Leave-One-Out Ablation
==========================================================
This is the ONLY feature selection stage.

Remove one feature at a time, retrain XGBoost_B, measure:
  Δ Spearman, Δ Pearson, Δ MAE, Δ RMSE, Δ R²

Rank all 11 features by marginal contribution.

Produces:
  results/exp2/plots/ablation/ablation_results.png
  stdout: ablation table
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    _TREE_ENGINE = "XGBoost"
except ImportError:
    try:
        from lightgbm import LGBMRegressor as XGBRegressor
        _TREE_ENGINE = "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
        _TREE_ENGINE = "HistGradientBoosting"

from config import (
    ALL_FEATURES,
    NEW_DESCRIPTORS,
    TARGET,
    TRAIN_PARQUET,
    TEST_PARQUET,
    XGBOOST_PARAMS,
    LAYER_DEPTH_MAP,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    set_pub_style,
    save_fig,
)
from phase3_regression import evaluate


def build_variant_b(df: pd.DataFrame, features: list):
    """Build Variant B (features + Relative_Depth) from a DataFrame."""
    if "Relative_Depth" not in df.columns:
        df = df.copy()
        df["Relative_Depth"] = df["Layer"].map(LAYER_DEPTH_MAP)

    cols = list(features) + ["Relative_Depth"]
    return df[cols].values, cols


def run_ablation() -> pd.DataFrame:
    """Execute leave-one-out ablation for all features."""
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    # Baseline: full model (all 11 features + Relative_Depth = Variant B)
    X_train_full, cols_full = build_variant_b(train_df, ALL_FEATURES)
    X_test_full, _ = build_variant_b(test_df, ALL_FEATURES)

    model_full = XGBRegressor(**XGBOOST_PARAMS)
    model_full.fit(X_train_full, y_train)
    y_pred_full = model_full.predict(X_test_full)
    baseline = evaluate(y_test, y_pred_full)

    print(f"[Phase 5] Baseline (all features): "
          f"ρ={baseline['Spearman']:+.4f}, R²={baseline['R2']:.4f}")

    # Ablation: remove one feature at a time
    ablation_rows = []

    for feat in ALL_FEATURES:
        reduced_features = [f for f in ALL_FEATURES if f != feat]
        X_train_red, _ = build_variant_b(train_df, reduced_features)
        X_test_red, _ = build_variant_b(test_df, reduced_features)

        model_red = XGBRegressor(**XGBOOST_PARAMS)
        model_red.fit(X_train_red, y_train)
        y_pred_red = model_red.predict(X_test_red)
        reduced = evaluate(y_test, y_pred_red)

        row = {
            "Removed_Feature": feat,
            "Type": "New" if feat in NEW_DESCRIPTORS else "Original",
            "Spearman_Full": baseline["Spearman"],
            "Spearman_Reduced": reduced["Spearman"],
            "Delta_Spearman": baseline["Spearman"] - reduced["Spearman"],
            "Pearson_Full": baseline["Pearson"],
            "Pearson_Reduced": reduced["Pearson"],
            "Delta_Pearson": baseline["Pearson"] - reduced["Pearson"],
            "MAE_Full": baseline["MAE"],
            "MAE_Reduced": reduced["MAE"],
            "Delta_MAE": reduced["MAE"] - baseline["MAE"],
            "RMSE_Full": baseline["RMSE"],
            "RMSE_Reduced": reduced["RMSE"],
            "Delta_RMSE": reduced["RMSE"] - baseline["RMSE"],
            "R2_Full": baseline["R2"],
            "R2_Reduced": reduced["R2"],
            "Delta_R2": baseline["R2"] - reduced["R2"],
        }

        ablation_rows.append(row)
        print(f"  Removed {feat:30s} → Δρ = {row['Delta_Spearman']:+.4f}, "
              f"ΔR² = {row['Delta_R2']:+.4f}")

    return pd.DataFrame(ablation_rows).sort_values("Delta_Spearman", ascending=False)


def plot_ablation(ablation_df: pd.DataFrame) -> None:
    """Ablation bar chart showing Δ Spearman when each feature is removed."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = ablation_df.sort_values("Delta_Spearman", ascending=True)
    colors = ["#E64A19" if f in NEW_DESCRIPTORS else "#1976D2"
              for f in df["Removed_Feature"]]

    ax.barh(range(len(df)), df["Delta_Spearman"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Removed_Feature"], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ Spearman ρ (positive = feature is helpful)")
    ax.set_title("Leave-One-Out Ablation: Feature Marginal Contribution")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#E64A19", label="New CARE Descriptor"),
               Patch(facecolor="#1976D2", label="Original Feature")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "ablation_results", subdir="ablation")


def main():
    set_global_seed()
    ensure_dirs()
    set_pub_style()

    print("=" * 70)
    print("PHASE 5 — LEAVE-ONE-OUT ABLATION")
    print("=" * 70)

    ablation_df = run_ablation()

    print("\n--- Ablation Results (sorted by contribution) ---")
    display_cols = ["Removed_Feature", "Type", "Delta_Spearman",
                     "Delta_R2", "Delta_MAE"]
    print(ablation_df[display_cols].to_markdown(index=False, floatfmt=".4f"))

    plot_ablation(ablation_df)

    print("\n[Phase 5] Leave-One-Out Ablation complete.")
    return ablation_df


if __name__ == "__main__":
    main()
