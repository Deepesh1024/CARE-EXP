"""
CARE-MoE Experiment 2 — Phase 2: Descriptor Diagnostics
==========================================================
Evaluate every new descriptor independently. NO feature rejection.

Produces:
  results/exp2/feature_statistics.csv
  results/exp2/plots/descriptor_scatter/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, skew, kurtosis

from config import (
    TRAIN_PARQUET,
    TEST_PARQUET,
    ORIGINAL_FEATURES,
    NEW_DESCRIPTORS,
    ALL_FEATURES,
    TARGET,
    FEATURE_STATS_CSV,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_csv,
    set_pub_style,
    save_fig,
)


def compute_feature_statistics(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Compute comprehensive statistics for each feature."""
    rows = []
    for feat in features:
        vals = df[feat].values
        target = df[TARGET].values

        sp_rho, sp_p = spearmanr(vals, target)
        pe_r, pe_p = pearsonr(vals, target)

        rows.append({
            "Feature": feat,
            "Type": "New" if feat in NEW_DESCRIPTORS else "Original",
            "Mean": np.mean(vals),
            "Std": np.std(vals),
            "Min": np.min(vals),
            "Max": np.max(vals),
            "Median": np.median(vals),
            "Skewness": float(skew(vals)),
            "Kurtosis": float(kurtosis(vals)),
            "Spearman_rho": float(sp_rho),
            "Spearman_p": float(sp_p),
            "Pearson_r": float(pe_r),
            "Pearson_p": float(pe_p),
        })

    return pd.DataFrame(rows)


def plot_descriptor_scatter(df: pd.DataFrame, feature: str) -> None:
    """Scatter plot of descriptor vs Oracle_KL, colored by layer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"first": "#1976D2", "middle": "#388E3C", "last": "#E64A19"}

    for layer in sorted(df["Layer"].unique()):
        sub = df[df["Layer"] == layer]
        ax.scatter(sub[feature], sub[TARGET],
                   alpha=0.3, s=12, color=colors[layer], label=layer)

    # Annotate with Spearman
    sp, _ = spearmanr(df[feature], df[TARGET])
    ax.text(0.02, 0.98, f"Spearman ρ = {sp:+.4f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(feature)
    ax.set_ylabel("Oracle KL")
    ax.set_title(f"{feature} vs. Oracle KL Divergence")
    ax.legend(title="Layer")
    fig.tight_layout()
    save_fig(fig, f"scatter_{feature.lower()}", subdir="descriptor_scatter")


def plot_descriptor_distribution(df: pd.DataFrame, feature: str) -> None:
    """Distribution histogram stratified by layer."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    layers = sorted(df["Layer"].unique())
    colors = {"first": "#1976D2", "middle": "#388E3C", "last": "#E64A19"}

    for ax, layer in zip(axes, layers):
        sub = df[df["Layer"] == layer]
        ax.hist(sub[feature], bins=50, alpha=0.7,
                color=colors[layer], edgecolor="white", linewidth=0.5)
        ax.set_title(f"Layer: {layer}")
        ax.set_xlabel(feature)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"{feature} Distribution by Layer", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"dist_{feature.lower()}", subdir="descriptor_scatter")


def plot_cross_correlation(df: pd.DataFrame) -> None:
    """Heatmap of correlations between new descriptors and original features."""
    import seaborn as sns

    all_cols = ALL_FEATURES + [TARGET]
    corr = df[all_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Spearman Correlation: All Features + Oracle KL", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "full_correlation_heatmap", subdir="descriptor_scatter")


def main():
    set_global_seed()
    ensure_dirs()
    set_pub_style()

    print("=" * 70)
    print("PHASE 2 — DESCRIPTOR DIAGNOSTICS")
    print("=" * 70)

    # Load augmented data (use combined train+test for diagnostics)
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"[Phase 2] Analyzing {len(full_df)} samples, "
          f"{len(ALL_FEATURES)} features")

    # Compute statistics
    stats_df = compute_feature_statistics(full_df, ALL_FEATURES)
    print("\n--- Feature Statistics ---")
    print(stats_df[["Feature", "Type", "Spearman_rho", "Pearson_r",
                     "Mean", "Std"]].to_markdown(index=False, floatfmt=".4f"))

    # Generate scatter plots for new descriptors
    for feat in NEW_DESCRIPTORS:
        plot_descriptor_scatter(full_df, feat)
        plot_descriptor_distribution(full_df, feat)

    # Cross-correlation heatmap
    plot_cross_correlation(full_df)

    # Save statistics
    save_csv(stats_df, FEATURE_STATS_CSV)

    print("\n[Phase 2] Descriptor Diagnostics complete.")
    return stats_df


if __name__ == "__main__":
    main()
