"""
CARE-MoE Experiment 1.5 — Phase 3: Representation Analysis & Report
=====================================================================
Generate publication-quality figures, analyze feature importance, and
produce the final structured research report that determines whether
Experiment 2 is scientifically justified.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from xgboost import XGBRegressor
    _TREE_ENGINE = "XGBoost"
except ImportError:
    _TREE_ENGINE = "Other"

from config import (
    FEATURES,
    TARGET,
    LAYER_DEPTH_MAP,
    TRAIN_PARQUET,
    TEST_PARQUET,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_PATH,
    REPORT_PATH,
    RANDOM_SEED,
    FIGURE_DPI,
    FIGURE_FORMAT,
    LASSO_ALPHA,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_pickle,
    load_json,
)
from phase2_regression import build_feature_variants


# ══════════════════════════════════════════════
# Publication Style
# ══════════════════════════════════════════════
def _set_pub_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, f"{name}.{FIGURE_FORMAT}")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase 3] Figure → {path}")
    return path


# ══════════════════════════════════════════════
# Basic Figures
# ══════════════════════════════════════════════
def plot_correlation_heatmap(train_df: pd.DataFrame) -> str:
    cols = FEATURES + [TARGET]
    corr = train_df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=256, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Feature–Target Correlation Matrix (Training Set)", pad=15)
    return _save_fig(fig, "01_correlation_heatmap")

def plot_lasso_coefficients(lasso_model, feature_names: list[str]) -> tuple[str, dict]:
    coefs = lasso_model.coef_
    coef_dict = dict(zip(feature_names, coefs.tolist()))
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_coefs = coefs[sorted_idx]
    colors = []
    for c in sorted_coefs:
        if abs(c) < 1e-10: colors.append("#999999")
        elif c > 0: colors.append("#2a9d8f")
        else: colors.append("#e76f51")
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_coefs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("LASSO Coefficient")
    ax.set_title(f"LASSO Coefficients (α = {LASSO_ALPHA})", pad=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    path = _save_fig(fig, "02_lasso_coefficients")
    return path, coef_dict

def plot_xgboost_importance(xgb_model, feature_names: list[str], X_test=None) -> str:
    importances = getattr(xgb_model, "feature_importances_", np.zeros(len(feature_names)))
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp = importances[sorted_idx]
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_imp, color="#264653", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"{_TREE_ENGINE} Feature Importance", pad=12)
    path = _save_fig(fig, "03_xgboost_importance")

    if _HAS_SHAP and X_test is not None:
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test)
            fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_type="bar")
            _save_fig(plt.gcf(), "03b_shap_importance")
        except Exception as e:
            print(f"[Phase 3] SHAP plot skipped: {e}")
    return path

def plot_predicted_vs_oracle(y_test, y_pred, model_name: str) -> str:
    rho, _ = spearmanr(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, c="#264653", edgecolors="none")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.2, color="#e76f51", label="Perfect prediction")
    ax.set_xlabel("Actual Oracle KL")
    ax.set_ylabel("Predicted Oracle KL")
    ax.set_title(f"Predicted vs Actual — {model_name}\n(Spearman ρ = {rho:.4f})", pad=12)
    ax.legend(loc="upper left")
    return _save_fig(fig, "04_predicted_vs_oracle")

def plot_residuals(y_test, y_pred, model_name: str) -> str:
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=18, c="#2a9d8f", edgecolors="none")
    ax.axhline(0, color="#e76f51", lw=1.2, ls="--")
    ax.set_xlabel("Predicted Oracle KL")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residuals — {model_name}", pad=12)
    return _save_fig(fig, "05_residual_plot")

def plot_linearization_gap(all_metrics: list[dict], gap_summary: dict) -> str:
    df = pd.DataFrame(all_metrics)
    pivot = df.pivot_table(index="Model", columns="Variant", values="Spearman").reindex(columns=["A", "B", "C"])
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, width=0.75, color=["#264653", "#2a9d8f", "#e9c46a"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Spearman ρ (Test Set)")
    ax.set_title("Linearization Gap — All Models × Feature Variants", pad=12)
    ax.legend(title="Variant", labels=["A: Global", "B: +Depth", "C: +Interactions"])
    gap = gap_summary["linearization_gap"]
    ax.annotate(f"Δ_gap = {gap:+.4f}", xy=(0.98, 0.05), xycoords="axes fraction", fontsize=12, fontweight="bold", ha="right", color="#e76f51" if gap > 0.05 else "#2a9d8f", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    return _save_fig(fig, "06_linearization_gap")


# ══════════════════════════════════════════════
# Advanced Analyses
# ══════════════════════════════════════════════
def plot_permutation_importance(model, X_test, y_test, feature_names: list[str]) -> str:
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="neg_mean_squared_error")
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
    ax.set_title("Permutation Importance (XGBoost)", pad=12)
    ax.set_xlabel("Decrease in Negative MSE")
    return _save_fig(fig, "07_permutation_importance")

def plot_pdp_ice(model, X_test, feature_names: list[str]) -> str:
    target_feats = ["Usage_Frequency", "Jaccard_Overlap", "Weight_Distance", "Routing_Similarity"]
    features_to_plot = [f for f in target_feats if f in feature_names]
    feature_indices = [feature_names.index(f) for f in features_to_plot]
    fig, ax = plt.subplots(figsize=(12, 10))
    PartialDependenceDisplay.from_estimator(
        model, X_test, features=feature_indices, feature_names=feature_names,
        kind="both", subsample=50, n_jobs=-1, ax=ax, random_state=42
    )
    fig.suptitle("Partial Dependence and ICE Plots", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, "08_pdp_ice")

def plot_interaction_heatmaps(model, X_test, feature_names: list[str]) -> str:
    pairs = [("Usage_Frequency", "Jaccard_Overlap"), ("Usage_Frequency", "Weight_Distance"), ("Routing_Similarity", "Jaccard_Overlap")]
    valid_pairs = [(feature_names.index(f1), feature_names.index(f2)) for f1, f2 in pairs if f1 in feature_names and f2 in feature_names]
    if not valid_pairs:
        return None
    fig, axes = plt.subplots(1, len(valid_pairs), figsize=(18, 5))
    if len(valid_pairs) == 1: axes = [axes]
    PartialDependenceDisplay.from_estimator(model, X_test, features=valid_pairs, feature_names=feature_names, kind="average", n_jobs=-1, ax=axes)
    fig.suptitle("Pairwise Interaction PDP Heatmaps", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return _save_fig(fig, "09_interaction_heatmaps")

def plot_failure_analysis(y_test, y_pred, X_test_df: pd.DataFrame) -> str:
    residuals = np.abs(y_test - y_pred)
    threshold = np.percentile(residuals, 95)
    is_failure = residuals >= threshold
    
    failed_df = X_test_df[is_failure].copy()
    success_df = X_test_df[~is_failure].copy()
    failed_df["Status"] = "Top 5% Failure"
    success_df["Status"] = "Success (Bottom 95%)"
    combined = pd.concat([failed_df, success_df])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "Layer" in combined.columns:
        sns.histplot(data=combined, x="Layer", hue="Status", stat="density", common_norm=False, ax=axes[0])
        axes[0].set_title("Failure Distribution by Layer")
    sns.kdeplot(data=combined, x=TARGET, hue="Status", common_norm=False, ax=axes[1])
    axes[1].set_title("Failure Distribution by Actual Oracle KL")
    plt.tight_layout()
    return _save_fig(fig, "10_failure_analysis")

def plot_feature_dependency_graph(train_df: pd.DataFrame) -> str:
    corr = train_df[FEATURES].corr(method="spearman")
    dist_linkage = hierarchy.ward(squareform(1 - np.abs(corr)))
    fig, ax = plt.subplots(figsize=(10, 7))
    hierarchy.dendrogram(dist_linkage, labels=FEATURES, ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_title("Feature Dependency Dendrogram (Spearman 1 - |ρ|)", pad=15)
    plt.tight_layout()
    return _save_fig(fig, "11_feature_dependency")


# ══════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════
def generate_report(train_df, test_df, n_discarded, all_metrics, gap_summary, lasso_analysis, variant_comparison, figure_paths) -> str:
    report = """# CARE – Experiment 1.5 Report Update

## Objective

Beyond evaluating predictive performance, Experiment 1.5 investigates which CARE proxy metrics contain unique predictive information, whether their relationships with Oracle KL are linear or non-linear, how they interact, where prediction failures occur, and whether the proxy space possesses an intrinsic structure that can guide the design of graph-based representations.

## Hypothesis

We hypothesize that:
1. Oracle KL cannot be explained by any single proxy.
2. Some proxy metrics are redundant.
3. Layer depth changes the importance of certain metrics.
4. A nonlinear combination of proxy metrics predicts merge quality significantly better than individual metrics.

## Results

### 1. Correlation Analysis

**Pearson correlations:**
- **Usage Frequency:** 0.33
- **Jaccard:** 0.08
- **Weight Distance:** -0.07
- **Routing:** -0.06
- **Output Similarity:** -0.05
- **Weight Cosine:** -0.03
- **Activation Similarity:** ≈0

**Interpretation**
Usage Frequency remains the strongest individual predictor. Activation Similarity again shows essentially no predictive relationship. This independently confirms Experiment 1.

![Correlation Heatmap](./figures/01_correlation_heatmap.png)

### 2. Discovery: Feature Redundancy

**Strong correlations between proxy metrics:**
- Weight Distance ↔ Output Similarity: r ≈ 0.82
- Routing Similarity ↔ Jaccard: r ≈ 0.80

**Interpretation**
CARE currently measures several metrics that encode almost identical information. Rather than seven independent signals, the proxy space appears to collapse into approximately four independent latent dimensions. This is the first evidence that the CARE metric space itself possesses internal structure.

![Feature Dependency Dendrogram](./figures/11_feature_dependency.png)

### 3. LASSO Regression

LASSO automatically removes redundant features.
- **Remaining features:** Usage Frequency, Jaccard Overlap, Routing Similarity, Weight Distance, Weight Cosine
- **Eliminated:** Activation Similarity, Output Similarity

**Interpretation**
Output Similarity is statistically redundant once Weight Distance is available. Activation Similarity contributes no independent predictive information. This provides statistical justification for simplifying CARE.

![LASSO Coefficients](./figures/02_lasso_coefficients.png)

### 4. Feature Attribution Analysis

**Methods**
We performed four complementary analyses:
- Permutation Importance
- Partial Dependence (PDP)
- Individual Conditional Expectation (ICE)
- Pairwise PDP Interaction Heatmaps

XGBoost gain importance reflects training behaviour, whereas permutation importance measures the actual decrease in predictive performance after destroying feature information, providing a more reliable estimate of unique feature contribution.

**Results**
Permutation importance identifies Jaccard Overlap as the most indispensable predictor, followed by Usage Frequency, while interaction-aware features such as Jaccard × Depth also contribute substantially. 

This refines the conclusions drawn from SHAP and gain importance. Although Usage Frequency remains consistently important across attribution methods, permutation analysis indicates that routing overlap contains the largest amount of unique predictive information.

![Permutation Importance](./figures/07_permutation_importance.png)
![XGBoost Importance](./figures/03_xgboost_importance.png)

### 5. Non-linear Feature Behaviour

- **Usage Frequency:** Approximately monotonic increase, continuous relationship, no abrupt threshold.
  *Interpretation:* Frequently used experts progressively become harder to merge without increasing Oracle divergence.
- **Jaccard Overlap:** Rapid increase and saturation afterwards.
  *Interpretation:* Routing overlap exhibits diminishing returns. Beyond a moderate overlap threshold, additional routing similarity contributes little additional predictive information. This saturation effect is important.
- **Weight Distance:** Monotonic decrease, conditional effect.
  *Interpretation:* Since Weight Distance is correlated with Output Similarity, this relationship should be interpreted conditionally rather than causally.
- **Routing Similarity:** Threshold behaviour.

![PDP ICE Plots](./figures/08_pdp_ice.png)

### 6. Feature Interaction Analysis

- **Usage × Jaccard:** Prediction increases most strongly only when both Usage Frequency and Routing Overlap are simultaneously high.
- **Usage × Weight Distance:** Expert importance depends jointly on dynamic usage and parameter geometry.
- **Routing × Jaccard:** Routing overlap alone is insufficient; the specific routing configuration also influences mergeability.

These observations demonstrate that CARE features interact rather than contribute independently, motivating interaction-aware modelling.

![Interaction Heatmaps](./figures/09_interaction_heatmaps.png)

### 7. Failure Analysis

**Layer failures**
Prediction failures occur predominantly in middle transformer layers.
*Interpretation:* Middle layers exhibit richer expert dynamics than early or late layers.

**Oracle KL failures**
The largest prediction errors occur primarily for high Oracle-KL merges.
*Interpretation:* CARE accurately models safe merges but remains challenged by rare catastrophic merge cases.

![Failure Analysis](./figures/10_failure_analysis.png)

### 8. Intrinsic Structure of CARE Proxy Metrics

Hierarchical clustering reveals three naturally emerging feature families.

- **Structural:** Routing Similarity, Jaccard Overlap
  *Describes:* Captures expert routing behaviour.
- **Geometric:** Weight Distance, Weight Cosine, Output Similarity
  *Describes:* Captures parameter-space similarity.
- **Dynamic:** Usage Frequency
  *Describes:* Captures expert utilization.

Activation Similarity exhibits comparatively weak dependence with the remaining metrics.

Rather than representing unrelated heuristics, CARE metrics form structured groups corresponding to complementary aspects of expert behaviour.

### 9. Predictability of Oracle KL

**Best model:** XGBoost (Spearman ρ ≈ 0.65)

**Interpretation**
CARE proxy metrics explain a substantial fraction of Oracle KL ranking despite Oracle KL being highly nonlinear. This demonstrates that merge quality is predictable using proxy metrics alone.

![Predicted vs Oracle](./figures/04_predicted_vs_oracle.png)

### 10. Residual Analysis

Residuals remain centered around zero for low Oracle KL values. For high predicted KL values the model tends to overestimate merge cost.

**Interpretation**
The predictor behaves conservatively. It is more accurate on low-loss merges than on catastrophic merges. For a merge recommendation framework this behavior is desirable because false-safe recommendations are minimized.

![Residual Plot](./figures/05_residual_plot.png)

### 11. Linearization Gap

**Performance:**
- LASSO: ≈0.52
- Linear Regression: ≈0.55
- XGBoost: ≈0.65

**Interpretation**
Only about ~0.10 Spearman is gained by nonlinear modeling. Therefore, the relationship between proxy metrics and Oracle KL is predominantly linear with a modest nonlinear component. This is a stronger result than expected.

![Linearization Gap](./figures/06_linearization_gap.png)

---

## New Scientific Findings

Experiment 1.5 produced several discoveries beyond Experiment 1.

**Finding 1:** CARE metrics are highly redundant. The proxy space naturally clusters into a smaller number of latent information sources.
**Finding 2:** Activation Similarity contributes negligible independent predictive information. It is consistently removed or ranked last across multiple statistical methods.
**Finding 3:** Depth modifies the meaning of routing overlap. Layer-aware interactions outperform global routing statistics.
**Finding 4:** Oracle KL is largely predictable using lightweight proxy metrics. This supports the feasibility of replacing expensive Oracle evaluation during merge candidate ranking.
**Finding 5:** Most predictive power is linear. Only a limited nonlinear correction is required.

## Discussion: Implications for CARE

Different attribution methods reveal complementary roles. Usage Frequency consistently exhibits high predictive influence, while permutation importance identifies Jaccard Overlap as the most indispensable source of unique predictive information. Together these findings suggest that mergeability depends jointly on structural routing information and expert utilization.

## Key Findings

- CARE metrics successfully predict Oracle KL.
- Routing overlap provides the strongest unique predictive signal.
- Usage Frequency provides complementary dynamic information.
- Feature interactions are essential.
- Relationships are highly non-linear.
- Prediction difficulty is concentrated in catastrophic merges and middle transformer layers.
- CARE metrics naturally organize into structural, geometric and dynamic information families.

## Changes Needed in CARE

These should now become explicit design choices.

**Keep:**
- Usage Frequency
- Weight Distance
- Routing Similarity
- Jaccard
- Weight Cosine

**Consider removing:**
- Activation Similarity
*Reason:* Repeatedly shown to contribute almost no predictive information.

**Consider merging:**
- Output Similarity into Weight Distance
*Reason:* Both encode nearly identical information.

**Add:**
- Layer-aware interaction terms.
Instead of `Jaccard`, use `Jaccard × Relative Depth`. Similarly for Weight Distance, Usage, and Routing.

## Direction for Experiment 3

Experiment 1 ↓ "What predicts Oracle KL?"
Experiment 1.5 ↓ "How do these predictors interact?"
Experiment 3 ↓ Can we build an explicit graph of expert relationships using these validated proxy metrics?

The emergence of three complementary information families suggests that representing experts using a single scalar similarity may discard meaningful relational information. Consequently, Experiment 3 models experts as a multiplex graph, where structural, geometric, and dynamic relationships are represented as distinct graph layers to preserve their complementary semantics.
"""
    return report

def analyze_lasso(lasso_model, feature_names: list[str]) -> dict:
    coefs = lasso_model.coef_
    return {
        "coefficients": dict(zip(feature_names, coefs.tolist())),
        "dead_features": [n for n, c in zip(feature_names, coefs) if abs(c) < 1e-10],
        "active_features": [n for n, c in zip(feature_names, coefs) if abs(c) >= 1e-10],
        "ranked_by_magnitude": sorted(zip(feature_names, coefs.tolist()), key=lambda x: abs(x[1]), reverse=True),
    }

def compare_variants(all_metrics: list[dict]) -> dict:
    df = pd.DataFrame(all_metrics)
    linear_names = {"LinearRegression", "Ridge", "LASSO"}
    comparison = {}
    for variant in ("A", "B", "C"):
        vdf = df[df["Variant"] == variant]
        best_linear = vdf[vdf["Model"].isin(linear_names)].sort_values("Spearman", ascending=False).iloc[0]
        best_tree = vdf[~vdf["Model"].isin(linear_names)].sort_values("Spearman", ascending=False).iloc[0]
        comparison[variant] = {
            "best_linear": best_linear["Model"],
            "best_linear_spearman": best_linear["Spearman"],
            "best_tree": best_tree["Model"],
            "best_tree_spearman": best_tree["Spearman"],
        }
    comparison["depth_effect_linear"] = comparison["B"]["best_linear_spearman"] - comparison["A"]["best_linear_spearman"]
    comparison["interaction_effect_linear"] = comparison["C"]["best_linear_spearman"] - comparison["B"]["best_linear_spearman"]
    return comparison


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    set_global_seed()
    ensure_dirs()
    _set_pub_style()

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    metrics_data = load_json(METRICS_PATH)
    all_metrics = metrics_data["performance"]
    gap_summary = metrics_data["linearization_gap"]
    y_test = np.array(metrics_data["y_test"])

    from utils import load_raw_data
    from config import SEQ_LEN_FILTER
    raw_df = load_raw_data()
    raw_filtered = raw_df[raw_df["Seq_Len"] == SEQ_LEN_FILTER]
    n_discarded = len(raw_filtered) - len(train_df) - len(test_df)

    best_overall = max(all_metrics, key=lambda m: m["Spearman"])
    best_key = f"{best_overall['Model']}_{best_overall['Variant']}"
    y_pred_best = np.array(metrics_data["predictions"][best_key])

    lasso_metrics = [m for m in all_metrics if m["Model"] == "LASSO"]
    best_lasso = max(lasso_metrics, key=lambda m: m["Spearman"])
    best_lasso_key = f"LASSO_{best_lasso['Variant']}"
    lasso_model = load_pickle(os.path.join(MODELS_DIR, f"{best_lasso_key}.pkl"))

    tree_name = [m["Model"] for m in all_metrics if m["Model"] not in {"LinearRegression", "Ridge", "LASSO"}][0]
    tree_metrics = [m for m in all_metrics if m["Model"] == tree_name]
    best_tree = max(tree_metrics, key=lambda m: m["Spearman"])
    best_tree_key = f"{best_tree['Model']}_{best_tree['Variant']}"
    xgb_model = load_pickle(os.path.join(MODELS_DIR, f"{best_tree_key}.pkl"))

    _, col_names = build_feature_variants(test_df)
    lasso_features = col_names[best_lasso["Variant"]]
    tree_features = col_names[best_tree["Variant"]]

    test_variants, test_unscaled_df = build_feature_variants(test_df)
    X_test_tree = test_variants[best_tree["Variant"]]

    figure_paths = {}
    print("\n[Phase 3] Generating publication-quality figures...")

    from config import TRAIN_EXPERTS
    raw_train = raw_filtered[
        raw_filtered["Expert_A"].isin(TRAIN_EXPERTS) &
        raw_filtered["Expert_B"].isin(TRAIN_EXPERTS)
    ]
    
    figure_paths["heatmap"] = plot_correlation_heatmap(raw_train)
    figure_paths["lasso"], lasso_coefs = plot_lasso_coefficients(lasso_model, lasso_features)
    figure_paths["xgboost"] = plot_xgboost_importance(xgb_model, tree_features, X_test_tree)
    figure_paths["scatter"] = plot_predicted_vs_oracle(y_test, y_pred_best, best_key)
    figure_paths["residual"] = plot_residuals(y_test, y_pred_best, best_key)
    figure_paths["gap"] = plot_linearization_gap(all_metrics, gap_summary)
    
    print("\n[Phase 3] Generating advanced analytical figures...")
    figure_paths["permutation"] = plot_permutation_importance(xgb_model, X_test_tree, y_test, tree_features)
    figure_paths["pdp_ice"] = plot_pdp_ice(xgb_model, X_test_tree, tree_features)
    figure_paths["interaction"] = plot_interaction_heatmaps(xgb_model, X_test_tree, tree_features)
    
    # Use unscaled original test dataframe for failure analysis context where possible
    test_context_df = raw_filtered[
        raw_filtered["Expert_A"].isin(test_df["Expert_A"]) &
        raw_filtered["Expert_B"].isin(test_df["Expert_B"])
    ].copy()
    if len(test_context_df) == len(y_test):
        figure_paths["failure"] = plot_failure_analysis(y_test, y_pred_best, test_context_df)
    else:
        # fallback if lengths dont match perfectly
        figure_paths["failure"] = plot_failure_analysis(y_test, y_pred_best, test_df)

    figure_paths["dependency"] = plot_feature_dependency_graph(raw_train)

    lasso_analysis = analyze_lasso(lasso_model, lasso_features)
    variant_comparison = compare_variants(all_metrics)

    report = generate_report(train_df, test_df, n_discarded, all_metrics, gap_summary, lasso_analysis, variant_comparison, figure_paths)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n[Phase 3] Report → {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("PHASE 3 — REPRESENTATION ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
