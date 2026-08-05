"""
CARE-MoE Experiment 2 — Phase 0.75: Existing Feature Correlation
=================================================================
Compute Pearson, Spearman, VIF, and multicollinearity diagnostics
for the 7 original pre-merge features.

Produces:
  results/exp2/correlation_matrix.csv
  results/exp2/plots/correlations/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from config import (
    EXP15_TRAIN_PARQUET,
    EXP15_TEST_PARQUET,
    ORIGINAL_FEATURES,
    TARGET,
    CORRELATION_MATRIX_CSV,
    PLOT_CORRELATIONS_DIR,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_csv,
    set_pub_style,
    save_fig,
)


def compute_vif(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R²_j), where R²_j is the R² of regressing feature j
    on all other features.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    feature_names : list of str

    Returns
    -------
    pd.DataFrame with columns: Feature, VIF
    """
    from sklearn.linear_model import LinearRegression

    vif_values = []
    n_features = X.shape[1]

    for j in range(n_features):
        y_j = X[:, j]
        X_other = np.delete(X, j, axis=1)

        reg = LinearRegression().fit(X_other, y_j)
        r2 = reg.score(X_other, y_j)
        vif = 1.0 / max(1.0 - r2, 1e-10)
        vif_values.append(vif)

    return pd.DataFrame({
        "Feature": feature_names,
        "VIF": vif_values,
    })


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              title: str, filename: str) -> None:
    """Generate a publication-quality correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))

    mask = np.zeros_like(corr_matrix, dtype=bool)
    # No mask — show full matrix

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": title.split(" ")[0]},
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, filename, subdir="correlations")


def plot_vif_bar(vif_df: pd.DataFrame) -> None:
    """Bar chart of VIF values with threshold lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    vif_sorted = vif_df.sort_values("VIF", ascending=True)
    colors = ["#E64A19" if v > 5 else "#1976D2" for v in vif_sorted["VIF"]]

    ax.barh(range(len(vif_sorted)), vif_sorted["VIF"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(vif_sorted)))
    ax.set_yticklabels(vif_sorted["Feature"], fontsize=10)
    ax.axvline(5, color="orange", linestyle="--", linewidth=1.5,
               label="VIF = 5 (moderate)")
    ax.axvline(10, color="red", linestyle="--", linewidth=1.5,
               label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("Multicollinearity: VIF per Feature")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, "vif_bar", subdir="correlations")


def main():
    set_global_seed()
    ensure_dirs()
    set_pub_style()

    print("=" * 70)
    print("PHASE 0.75 — EXISTING FEATURE CORRELATION")
    print("=" * 70)

    # Load both train and test to compute correlations on full available data
    train_df = pd.read_parquet(EXP15_TRAIN_PARQUET)
    test_df = pd.read_parquet(EXP15_TEST_PARQUET)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    cols = ORIGINAL_FEATURES + [TARGET]
    data = full_df[cols]

    print(f"[Phase 0.75] Computing correlations on {len(data)} samples, "
          f"{len(cols)} columns")

    # Pearson correlation
    pearson_corr = data.corr(method="pearson")
    print("\n--- Pearson Correlation Matrix ---")
    print(pearson_corr.to_string(float_format="%.3f"))

    # Spearman correlation
    spearman_corr = data.corr(method="spearman")
    print("\n--- Spearman Correlation Matrix ---")
    print(spearman_corr.to_string(float_format="%.3f"))

    # VIF (on features only, not target)
    X = full_df[ORIGINAL_FEATURES].values
    vif_df = compute_vif(X, ORIGINAL_FEATURES)
    print("\n--- Variance Inflation Factors ---")
    print(vif_df.to_markdown(index=False, floatfmt=".3f"))

    # Multicollinearity interpretation
    print("\n--- Multicollinearity Assessment ---")
    high_vif = vif_df[vif_df["VIF"] > 5]
    if len(high_vif) > 0:
        print(f"  Features with VIF > 5 (moderate multicollinearity):")
        for _, row in high_vif.iterrows():
            print(f"    {row['Feature']}: VIF = {row['VIF']:.3f}")
    else:
        print("  No features exceed VIF = 5 threshold.")

    severe_vif = vif_df[vif_df["VIF"] > 10]
    if len(severe_vif) > 0:
        print(f"  Features with VIF > 10 (severe multicollinearity):")
        for _, row in severe_vif.iterrows():
            print(f"    {row['Feature']}: VIF = {row['VIF']:.3f}")

    # Generate plots
    plot_correlation_heatmap(pearson_corr, "Pearson Correlation Matrix",
                             "pearson_heatmap")
    plot_correlation_heatmap(spearman_corr, "Spearman Correlation Matrix",
                             "spearman_heatmap")
    plot_vif_bar(vif_df)

    # Save combined correlation data
    combined = pd.DataFrame({
        "Feature": ORIGINAL_FEATURES,
        "Pearson_with_Oracle": [pearson_corr.loc[f, TARGET]
                                 for f in ORIGINAL_FEATURES],
        "Spearman_with_Oracle": [spearman_corr.loc[f, TARGET]
                                  for f in ORIGINAL_FEATURES],
        "VIF": vif_df.set_index("Feature").loc[ORIGINAL_FEATURES, "VIF"].values,
    })
    save_csv(combined, CORRELATION_MATRIX_CSV)

    print("\n[Phase 0.75] Feature Correlation Analysis complete.")
    return pearson_corr, spearman_corr, vif_df


if __name__ == "__main__":
    main()
