"""
CARE-MoE Experiment 2 — Phase 4: Interpretability
====================================================
Generate LASSO coefficients, SHAP, XGBoost gain importance,
and permutation importance.

Produces:
  results/exp2/feature_importance.csv
  results/exp2/plots/shap/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from config import (
    ALL_FEATURES,
    NEW_DESCRIPTORS,
    TARGET,
    TRAIN_PARQUET,
    TEST_PARQUET,
    MODELS_DIR,
    FEATURE_IMPORTANCE_CSV,
    LAYER_DEPTH_MAP,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_pickle,
    save_csv,
    set_pub_style,
    save_fig,
)
from phase3_regression import build_feature_variants


def get_lasso_coefficients(variant: str = "A") -> pd.DataFrame:
    """Extract LASSO coefficients for the specified variant."""
    model = load_pickle(os.path.join(MODELS_DIR, f"LASSO_{variant}.pkl"))

    if variant == "A":
        cols = list(ALL_FEATURES)
    elif variant == "B":
        cols = list(ALL_FEATURES) + ["Relative_Depth"]
    else:
        cols = (list(ALL_FEATURES) + ["Relative_Depth"] +
                [f"{f}_x_depth" for f in ALL_FEATURES])

    coefs = model.coef_
    return pd.DataFrame({
        "Feature": cols[:len(coefs)],
        "LASSO_Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs),
    }).sort_values("Abs_Coefficient", ascending=False)


def plot_lasso_coefficients(coef_df: pd.DataFrame) -> None:
    """Bar chart of LASSO coefficients."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#E64A19" if f in NEW_DESCRIPTORS else "#1976D2"
              for f in coef_df["Feature"]]

    ax.barh(range(len(coef_df)), coef_df["LASSO_Coefficient"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df["Feature"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("LASSO Coefficient")
    ax.set_title("LASSO_A Coefficient Profile (Augmented Features)")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#E64A19", label="New CARE Descriptor"),
               Patch(facecolor="#1976D2", label="Original Feature")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "lasso_coefficients", subdir="shap")


def get_xgboost_importance(variant: str = "B") -> pd.DataFrame:
    """Extract XGBoost gain-based feature importance."""
    model = load_pickle(os.path.join(MODELS_DIR, f"XGBoost_{variant}.pkl"))

    if variant == "A":
        cols = list(ALL_FEATURES)
    elif variant == "B":
        cols = list(ALL_FEATURES) + ["Relative_Depth"]
    else:
        cols = (list(ALL_FEATURES) + ["Relative_Depth"] +
                [f"{f}_x_depth" for f in ALL_FEATURES])

    importances = model.feature_importances_
    return pd.DataFrame({
        "Feature": cols[:len(importances)],
        "XGBoost_Gain": importances,
    }).sort_values("XGBoost_Gain", ascending=False)


def plot_xgboost_importance(imp_df: pd.DataFrame) -> None:
    """Bar chart of XGBoost gain importance."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#E64A19" if f in NEW_DESCRIPTORS else "#1976D2"
              for f in imp_df["Feature"]]

    ax.barh(range(len(imp_df)), imp_df["XGBoost_Gain"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["Feature"], fontsize=9)
    ax.set_xlabel("Gain Importance")
    ax.set_title("XGBoost_B Feature Importance (Augmented Features)")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#E64A19", label="New CARE Descriptor"),
               Patch(facecolor="#1976D2", label="Original Feature")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "xgboost_importance", subdir="shap")


def compute_shap_values(variant: str = "B"):
    """Compute SHAP values for XGBoost model."""
    if not _HAS_SHAP:
        print("[Phase 4] SHAP not available — skipping SHAP analysis.")
        return None, None

    model = load_pickle(os.path.join(MODELS_DIR, f"XGBoost_{variant}.pkl"))
    test_df = pd.read_parquet(TEST_PARQUET)

    test_variants, test_cols = build_feature_variants(test_df)
    X_test = test_variants[variant]
    cols = test_cols[variant]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return shap_values, cols


def plot_shap_summary(shap_values, feature_names, variant="B"):
    """SHAP beeswarm summary plot."""
    if shap_values is None:
        return

    test_df = pd.read_parquet(TEST_PARQUET)
    test_variants, _ = build_feature_variants(test_df)
    X_test = test_variants[variant]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                       show=False, max_display=15)
    plt.title("SHAP Summary — XGBoost_B (Augmented Features)")
    plt.tight_layout()
    save_fig(plt.gcf(), "shap_summary", subdir="shap")


def compute_permutation_importance(variant: str = "B"):
    """Compute permutation importance for XGBoost."""
    from sklearn.inspection import permutation_importance

    model = load_pickle(os.path.join(MODELS_DIR, f"XGBoost_{variant}.pkl"))
    test_df = pd.read_parquet(TEST_PARQUET)
    y_test = test_df[TARGET].values

    test_variants, test_cols = build_feature_variants(test_df)
    X_test = test_variants[variant]
    cols = test_cols[variant]

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error"
    )

    perm_df = pd.DataFrame({
        "Feature": cols,
        "Perm_Importance_Mean": result.importances_mean,
        "Perm_Importance_Std": result.importances_std,
    }).sort_values("Perm_Importance_Mean", ascending=False)

    return perm_df


def plot_permutation_importance(perm_df: pd.DataFrame) -> None:
    """Bar chart of permutation importance."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#E64A19" if f in NEW_DESCRIPTORS else "#1976D2"
              for f in perm_df["Feature"]]

    ax.barh(range(len(perm_df)), perm_df["Perm_Importance_Mean"],
            xerr=perm_df["Perm_Importance_Std"],
            color=colors, edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(perm_df)))
    ax.set_yticklabels(perm_df["Feature"], fontsize=9)
    ax.set_xlabel("Permutation Importance (ΔMAE)")
    ax.set_title("Permutation Importance — XGBoost_B (Augmented)")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#E64A19", label="New CARE Descriptor"),
               Patch(facecolor="#1976D2", label="Original Feature")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "permutation_importance", subdir="shap")


def main():
    set_global_seed()
    ensure_dirs()
    set_pub_style()

    print("=" * 70)
    print("PHASE 4 — INTERPRETABILITY")
    print("=" * 70)

    # LASSO coefficients
    coef_df = get_lasso_coefficients("A")
    print("\n--- LASSO_A Coefficients ---")
    print(coef_df.to_markdown(index=False, floatfmt=".6f"))
    plot_lasso_coefficients(coef_df)

    # XGBoost importance
    imp_df = get_xgboost_importance("B")
    print("\n--- XGBoost_B Gain Importance ---")
    print(imp_df.to_markdown(index=False, floatfmt=".4f"))
    plot_xgboost_importance(imp_df)

    # SHAP
    shap_values, shap_cols = compute_shap_values("B")
    plot_shap_summary(shap_values, shap_cols)

    # Permutation importance
    perm_df = compute_permutation_importance("B")
    print("\n--- Permutation Importance (XGBoost_B) ---")
    print(perm_df.to_markdown(index=False, floatfmt=".6f"))
    plot_permutation_importance(perm_df)

    # Save combined importance table
    combined = coef_df[["Feature", "LASSO_Coefficient"]].merge(
        imp_df[["Feature", "XGBoost_Gain"]], on="Feature", how="outer"
    ).merge(
        perm_df[["Feature", "Perm_Importance_Mean"]], on="Feature", how="outer"
    )
    save_csv(combined, FEATURE_IMPORTANCE_CSV)

    print("\n[Phase 4] Interpretability Analysis complete.")
    return coef_df, imp_df, perm_df


if __name__ == "__main__":
    main()
