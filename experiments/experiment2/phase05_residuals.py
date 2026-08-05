"""
CARE-MoE Experiment 2 — Phase 0.5: Residual Analysis
=======================================================
Analyze failure modes of the Experiment 1.5 baseline to motivate
new capability descriptors.

Uses frozen predictions from metrics.json (XGBoost_B) and the
Exp 1.5 test parquet.

Produces:
  results/exp2/residual_analysis.csv
  results/exp2/plots/residuals/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    EXP15_METRICS_PATH,
    EXP15_TEST_PARQUET,
    ORIGINAL_FEATURES,
    TARGET,
    RESIDUAL_ANALYSIS_CSV,
    PLOT_RESIDUALS_DIR,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_json,
    save_csv,
    set_pub_style,
    save_fig,
)


def load_residuals() -> pd.DataFrame:
    """Load Exp 1.5 test data and frozen predictions, compute residuals.

    Returns
    -------
    pd.DataFrame : test data augmented with prediction and residual columns.
    """
    # Load frozen baseline predictions
    metrics = load_json(EXP15_METRICS_PATH)
    y_test = np.array(metrics["y_test"])
    pred_xgb_b = np.array(metrics["predictions"]["XGBoost_B"])
    pred_lr_a = np.array(metrics["predictions"]["LinearRegression_A"])

    # Load test parquet (has metadata columns: Layer, Expert_A, Expert_B, etc.)
    test_df = pd.read_parquet(EXP15_TEST_PARQUET)

    # Verify alignment
    assert len(test_df) == len(y_test), (
        f"Parquet rows ({len(test_df)}) != y_test length ({len(y_test)})"
    )

    # Add predictions and residuals
    test_df["Pred_XGBoost_B"] = pred_xgb_b
    test_df["Pred_LR_A"] = pred_lr_a
    test_df["Residual_XGBoost_B"] = test_df[TARGET] - pred_xgb_b
    test_df["Residual_LR_A"] = test_df[TARGET] - pred_lr_a
    test_df["AbsResidual_XGBoost_B"] = test_df["Residual_XGBoost_B"].abs()

    print(f"[Phase 0.5] Loaded {len(test_df)} test samples with residuals")
    return test_df


def plot_residual_by_layer(df: pd.DataFrame) -> None:
    """Residual distribution stratified by network layer."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    layers = sorted(df["Layer"].unique())

    for ax, layer in zip(axes, layers):
        sub = df[df["Layer"] == layer]
        ax.hist(sub["Residual_XGBoost_B"], bins=50, alpha=0.7,
                color="#2196F3", edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Layer: {layer} (n={len(sub)})")
        ax.set_xlabel("Residual (Oracle KL − Predicted)")

        # Annotate statistics
        mean_r = sub["Residual_XGBoost_B"].mean()
        std_r = sub["Residual_XGBoost_B"].std()
        ax.text(0.95, 0.95, f"μ={mean_r:.5f}\nσ={std_r:.5f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", alpha=0.8))

    axes[0].set_ylabel("Count")
    fig.suptitle("XGBoost_B Residual Distribution by Layer", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "residual_by_layer", subdir="residuals")


def plot_residual_vs_oracle(df: pd.DataFrame) -> None:
    """Residual magnitude vs actual Oracle KL (heteroscedasticity check)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"first": "#1976D2", "middle": "#388E3C", "last": "#E64A19"}

    for layer in sorted(df["Layer"].unique()):
        sub = df[df["Layer"] == layer]
        ax.scatter(sub[TARGET], sub["Residual_XGBoost_B"],
                   alpha=0.3, s=12, color=colors[layer], label=layer)

    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Actual Oracle KL")
    ax.set_ylabel("Residual (Oracle KL − Predicted)")
    ax.set_title("Heteroscedasticity: Residual vs. Oracle KL")
    ax.legend(title="Layer")
    fig.tight_layout()
    save_fig(fig, "residual_vs_oracle", subdir="residuals")


def plot_residual_vs_feature(df: pd.DataFrame, feature: str,
                              label: str = None) -> None:
    """Residual magnitude vs a specific feature, colored by layer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"first": "#1976D2", "middle": "#388E3C", "last": "#E64A19"}
    label = label or feature

    for layer in sorted(df["Layer"].unique()):
        sub = df[df["Layer"] == layer]
        ax.scatter(sub[feature], sub["AbsResidual_XGBoost_B"],
                   alpha=0.3, s=12, color=colors[layer], label=layer)

    ax.set_xlabel(label)
    ax.set_ylabel("|Residual| (XGBoost_B)")
    ax.set_title(f"Absolute Residual vs. {label}")
    ax.legend(title="Layer")
    fig.tight_layout()
    save_fig(fig, f"residual_vs_{feature.lower()}", subdir="residuals")


def plot_top_failures(df: pd.DataFrame, n: int = 20) -> None:
    """Bar chart of the top-N highest-residual expert pairs."""
    top = df.nlargest(n, "AbsResidual_XGBoost_B").copy()
    top["Pair"] = top.apply(
        lambda r: f"({int(r['Expert_A'])},{int(r['Expert_B'])})-{r['Layer']}", axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"first": "#1976D2", "middle": "#388E3C", "last": "#E64A19"}
    bar_colors = [colors[l] for l in top["Layer"]]

    ax.barh(range(len(top)), top["AbsResidual_XGBoost_B"],
            color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Pair"], fontsize=8)
    ax.set_xlabel("|Residual|")
    ax.set_title(f"Top {n} Highest-Residual Expert Pairs (XGBoost_B)")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    ax.legend(handles=handles, title="Layer", loc="lower right")

    fig.tight_layout()
    save_fig(fig, "top_failures", subdir="residuals")


def compute_failure_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-layer residual statistics and feature means for failures."""
    # Define "failure" as top 10% absolute residual
    threshold = df["AbsResidual_XGBoost_B"].quantile(0.90)
    df["Is_Failure"] = df["AbsResidual_XGBoost_B"] >= threshold

    stats_rows = []
    for layer in sorted(df["Layer"].unique()):
        sub = df[df["Layer"] == layer]
        failures = sub[sub["Is_Failure"]]
        successes = sub[~sub["Is_Failure"]]

        row = {
            "Layer": layer,
            "N_Total": len(sub),
            "N_Failures": len(failures),
            "Failure_Rate": len(failures) / max(len(sub), 1),
            "Mean_Residual": sub["Residual_XGBoost_B"].mean(),
            "Std_Residual": sub["Residual_XGBoost_B"].std(),
            "Mean_OracleKL_Failures": failures[TARGET].mean() if len(failures) > 0 else 0,
            "Mean_OracleKL_Successes": successes[TARGET].mean() if len(successes) > 0 else 0,
        }
        # Add mean feature values for failures vs successes
        for feat in ORIGINAL_FEATURES:
            if feat in sub.columns:
                row[f"MeanFail_{feat}"] = failures[feat].mean() if len(failures) > 0 else 0
                row[f"MeanSucc_{feat}"] = successes[feat].mean() if len(successes) > 0 else 0

        stats_rows.append(row)

    return pd.DataFrame(stats_rows)


def main():
    set_global_seed()
    ensure_dirs()
    set_pub_style()

    print("=" * 70)
    print("PHASE 0.5 — RESIDUAL ANALYSIS")
    print("=" * 70)

    df = load_residuals()

    # Generate plots
    plot_residual_by_layer(df)
    plot_residual_vs_oracle(df)
    plot_residual_vs_feature(df, "Routing_Similarity")
    plot_residual_vs_feature(df, "Usage_Frequency")
    plot_residual_vs_feature(df, "Weight_Distance")
    plot_top_failures(df)

    # Compute statistics
    stats_df = compute_failure_statistics(df)
    print("\n--- Failure Statistics by Layer ---")
    print(stats_df[["Layer", "N_Total", "N_Failures", "Failure_Rate",
                     "Mean_Residual", "Std_Residual"]].to_markdown(index=False))

    # Save full residual table
    export_cols = (
        ["Layer", "Expert_A", "Expert_B", TARGET,
         "Pred_XGBoost_B", "Residual_XGBoost_B", "AbsResidual_XGBoost_B"]
        + ORIGINAL_FEATURES
    )
    save_csv(df[export_cols], RESIDUAL_ANALYSIS_CSV)

    print("\n[Phase 0.5] Residual Analysis complete.")
    return df, stats_df


if __name__ == "__main__":
    main()
