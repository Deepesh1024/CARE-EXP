"""
CARE-MoE Experiment 1.5 — Phase 3: Representation Analysis & Report
=====================================================================
Generate publication-quality figures, analyze feature importance, and
produce the final structured research report that determines whether
Experiment 2 is scientifically justified.

Figures
-------
    1. Correlation heatmap (features + target)
    2. LASSO coefficient bar chart
    3. XGBoost feature importance (gain)
    4. Predicted vs Oracle scatter plot
    5. Residual plot
    6. Linearization Gap summary bar plot

Report
------
    results/exp1_5/experiment1_5_report.md
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
# Figure 1: Correlation Heatmap
# ══════════════════════════════════════════════
def plot_correlation_heatmap(train_df: pd.DataFrame) -> str:
    """Heatmap of Pearson correlations among features and Oracle_KL."""
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


# ══════════════════════════════════════════════
# Figure 2: LASSO Coefficient Bar Chart
# ══════════════════════════════════════════════
def plot_lasso_coefficients(lasso_model, feature_names: list[str]) -> tuple[str, dict]:
    """Bar chart of LASSO coefficients; returns figure path and coef dict."""
    coefs = lasso_model.coef_
    coef_dict = dict(zip(feature_names, coefs.tolist()))

    # Sort by absolute magnitude
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_coefs = coefs[sorted_idx]

    # Color: positive = teal, negative = coral, zero = gray
    colors = []
    for c in sorted_coefs:
        if abs(c) < 1e-10:
            colors.append("#999999")
        elif c > 0:
            colors.append("#2a9d8f")
        else:
            colors.append("#e76f51")

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_coefs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("LASSO Coefficient")
    ax.set_title(f"LASSO Coefficients (α = {LASSO_ALPHA})", pad=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    # Annotate zero coefficients
    dead_features = [n for n, c in zip(feature_names, coefs) if abs(c) < 1e-10]
    if dead_features:
        ax.annotate(
            f"{len(dead_features)} features eliminated (coef = 0)",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=9, fontstyle="italic", color="#666666",
        )

    path = _save_fig(fig, "02_lasso_coefficients")
    return path, coef_dict


# ══════════════════════════════════════════════
# Figure 3: XGBoost Feature Importance
# ══════════════════════════════════════════════
def plot_xgboost_importance(xgb_model, feature_names: list[str], X_test=None) -> str:
    """Bar chart of XGBoost feature importance (gain).

    If SHAP is available and X_test is provided, overlays SHAP values.
    """
    # Gain-based importance
    if hasattr(xgb_model, "feature_importances_"):
        importances = xgb_model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

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

    # ── Optional SHAP summary ──
    if _HAS_SHAP and X_test is not None:
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test)
            fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_values, X_test,
                feature_names=feature_names,
                show=False, plot_type="bar",
            )
            _save_fig(plt.gcf(), "03b_shap_importance")
        except Exception as e:
            print(f"[Phase 3] SHAP plot skipped: {e}")

    return path


# ══════════════════════════════════════════════
# Figure 4: Predicted vs Oracle Scatter
# ══════════════════════════════════════════════
def plot_predicted_vs_oracle(y_test, y_pred, model_name: str) -> str:
    """Scatter of predicted vs actual Oracle_KL with perfect-prediction line."""
    rho, _ = spearmanr(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, c="#264653", edgecolors="none")

    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", lw=1.2, color="#e76f51", label="Perfect prediction")

    ax.set_xlabel("Actual Oracle KL")
    ax.set_ylabel("Predicted Oracle KL")
    ax.set_title(f"Predicted vs Actual — {model_name}\n(Spearman ρ = {rho:.4f})", pad=12)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")

    return _save_fig(fig, "04_predicted_vs_oracle")


# ══════════════════════════════════════════════
# Figure 5: Residual Plot
# ══════════════════════════════════════════════
def plot_residuals(y_test, y_pred, model_name: str) -> str:
    """Residual plot: residuals vs predicted values."""
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=18, c="#2a9d8f", edgecolors="none")
    ax.axhline(0, color="#e76f51", lw=1.2, ls="--")
    ax.set_xlabel("Predicted Oracle KL")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residuals — {model_name}", pad=12)

    return _save_fig(fig, "05_residual_plot")


# ══════════════════════════════════════════════
# Figure 6: Linearization Gap Summary
# ══════════════════════════════════════════════
def plot_linearization_gap(all_metrics: list[dict], gap_summary: dict) -> str:
    """Grouped bar chart comparing Spearman ρ across all models and variants."""
    df = pd.DataFrame(all_metrics)

    # Pivot for grouped bars
    pivot = df.pivot_table(index="Model", columns="Variant", values="Spearman")
    pivot = pivot.reindex(columns=["A", "B", "C"])

    # Sort by best Spearman
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, width=0.75,
               color=["#264653", "#2a9d8f", "#e9c46a"],
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Spearman ρ (Test Set)")
    ax.set_title("Linearization Gap — All Models × Feature Variants", pad=12)
    ax.legend(title="Variant", labels=["A: Global", "B: +Depth", "C: +Interactions"])

    # Annotate the gap
    gap = gap_summary["linearization_gap"]
    ax.annotate(
        f"Δ_gap = {gap:+.4f}",
        xy=(0.98, 0.05), xycoords="axes fraction",
        fontsize=12, fontweight="bold", ha="right",
        color="#e76f51" if gap > 0.05 else "#2a9d8f",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    return _save_fig(fig, "06_linearization_gap")


# ══════════════════════════════════════════════
# LASSO Analysis
# ══════════════════════════════════════════════
def analyze_lasso(lasso_model, feature_names: list[str]) -> dict:
    """Return structured LASSO analysis: active, dead, ranked features."""
    coefs = lasso_model.coef_
    analysis = {
        "coefficients": dict(zip(feature_names, coefs.tolist())),
        "dead_features": [n for n, c in zip(feature_names, coefs) if abs(c) < 1e-10],
        "active_features": [n for n, c in zip(feature_names, coefs) if abs(c) >= 1e-10],
        "ranked_by_magnitude": sorted(
            zip(feature_names, coefs.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        ),
    }
    return analysis


# ══════════════════════════════════════════════
# Variant Comparison
# ══════════════════════════════════════════════
def compare_variants(all_metrics: list[dict]) -> dict:
    """Compare Models A, B, C to determine depth and interaction effects."""
    df = pd.DataFrame(all_metrics)
    linear_names = {"LinearRegression", "Ridge", "LASSO"}

    comparison = {}
    for variant in ("A", "B", "C"):
        vdf = df[df["Variant"] == variant]
        best_linear = vdf[vdf["Model"].isin(linear_names)].sort_values(
            "Spearman", ascending=False
        ).iloc[0]
        best_tree = vdf[~vdf["Model"].isin(linear_names)].sort_values(
            "Spearman", ascending=False
        ).iloc[0]
        comparison[variant] = {
            "best_linear": best_linear["Model"],
            "best_linear_spearman": best_linear["Spearman"],
            "best_tree": best_tree["Model"],
            "best_tree_spearman": best_tree["Spearman"],
        }

    # Depth effect: B vs A
    depth_effect = (
        comparison["B"]["best_linear_spearman"] - comparison["A"]["best_linear_spearman"]
    )
    # Interaction effect: C vs B
    interaction_effect = (
        comparison["C"]["best_linear_spearman"] - comparison["B"]["best_linear_spearman"]
    )

    comparison["depth_effect_linear"] = depth_effect
    comparison["interaction_effect_linear"] = interaction_effect

    return comparison


# ══════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════
def generate_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_discarded: int,
    all_metrics: list[dict],
    gap_summary: dict,
    lasso_analysis: dict,
    variant_comparison: dict,
    figure_paths: dict,
) -> str:
    """Generate the structured markdown research report."""

    perf_df = pd.DataFrame(all_metrics)[
        ["Model", "Variant", "N_Features", "Spearman", "Pearson", "MAE", "RMSE", "R2"]
    ].sort_values(["Variant", "Spearman"], ascending=[True, False])
    perf_table = perf_df.to_markdown(index=False, floatfmt=".4f")

    gap = gap_summary["linearization_gap"]
    best_lin = gap_summary["best_linear_model"]
    best_tree = gap_summary["best_tree_model"]
    rho_lin = gap_summary["best_linear_spearman"]
    rho_tree = gap_summary["best_tree_spearman"]

    dead = lasso_analysis["dead_features"]
    active = lasso_analysis["active_features"]
    ranked = lasso_analysis["ranked_by_magnitude"]

    depth_eff = variant_comparison["depth_effect_linear"]
    inter_eff = variant_comparison["interaction_effect_linear"]

    # ── Scientific interpretation ──
    gap_abs = abs(gap)
    if gap_abs < 0.03:
        gap_interpretation = (
            "The linearization gap is **negligible** (Δ < 0.03). "
            "The existing feature family captures essentially all the information "
            "available to the nonlinear model. No new features are needed."
        )
        outcome = "A"
        outcome_text = (
            "**Outcome A — Existing feature family is sufficient.**\n\n"
            "Experiment 2 is unnecessary. The current 13-feature family, when combined "
            "in a linear model, approaches the nonlinear ceiling. Proceed directly to "
            "optimization (Experiment 3)."
        )
    elif gap_abs < 0.10:
        gap_interpretation = (
            f"The linearization gap is **moderate** (Δ = {gap:.4f}). "
            "There is some nonlinear structure the linear model cannot capture, "
            "but the gap is small enough that the existing features carry most "
            "of the signal. Experiment 2 may yield marginal improvements."
        )
        outcome = "B (marginal)"
        outcome_text = (
            "**Outcome B (Marginal) — Experiment 2 is recommended but not critical.**\n\n"
            "The existing feature family captures most of the signal, but a moderate "
            "linearization gap suggests some nonlinear interactions remain unexploited. "
            "Experiment 2's objective: discover new pairwise features that shrink the "
            "Linearization Gap and enable a simple linear model to approach the "
            "nonlinear ceiling."
        )
    else:
        gap_interpretation = (
            f"The linearization gap is **large** (Δ = {gap:.4f}). "
            "The nonlinear model extracts substantially more signal than any linear "
            "combination of existing features. This indicates the feature family is "
            "insufficient for a linear predictor."
        )
        outcome = "B"
        outcome_text = (
            "**Outcome B — Existing feature family is insufficient.**\n\n"
            "Experiment 2 is required. Its objective: discover new pairwise features "
            "that shrink the Linearization Gap and enable a simple linear model to "
            "approach the nonlinear ceiling."
        )

    # ── Compose report ──
    report = f"""# CARE-MoE Experiment 1.5 — Linearization Gap Analysis

## Research Question

> Can the existing feature family, when combined, explain Oracle Capability Drift?

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Total pairs in dataset | {len(train_df) + len(test_df) + n_discarded:,} |
| Training pairs | {len(train_df):,} |
| Testing pairs | {len(test_df):,} |
| Discarded cross-boundary pairs | {n_discarded:,} |
| Number of base features | {len(active) + len(dead)} |
| Active features (LASSO) | {len(active)} |
| Dead features (LASSO) | {len(dead)} |
| Expert split | Train: 0–31, Test: 32–63 |
| Seq_Len filter | 256 |
| Layers | {sorted(train_df['Layer'].unique())} |
| Scaling | RobustScaler (fit on training set) |

---

## 2. Model Performance

{perf_table}

---

## 3. Linearization Gap

| Metric | Value |
|---|---|
| Best Linear Model | {best_lin} |
| Best Linear ρ | {rho_lin:.4f} |
| Best Tree Model | {best_tree} |
| Best Tree ρ | {rho_tree:.4f} |
| **Δ_gap** | **{gap:+.4f}** |

### Interpretation

{gap_interpretation}

---

## 4. Feature Analysis

### 4.1 LASSO Coefficients (α = {LASSO_ALPHA})

| Feature | Coefficient |
|---|---|
"""
    for name, coef in ranked:
        status = " ⚠️ DEAD" if abs(coef) < 1e-10 else ""
        report += f"| {name} | {coef:+.6f}{status} |\n"

    report += f"""
**Dead features** (coefficient = 0, mathematically eliminated by LASSO):
{', '.join(dead) if dead else 'None — all features retained.'}

**Active features** ({len(active)}):
{', '.join(active)}

### 4.2 Depth Effects

| Comparison | Δ Spearman |
|---|---|
| Model B (+ depth) vs Model A (global) | {depth_eff:+.4f} |
| Model C (+ interactions) vs Model B (+ depth) | {inter_eff:+.4f} |

"""
    if abs(depth_eff) > 0.01:
        report += "Adding relative layer depth **improves** prediction, confirming layer-dependent behavior.\n\n"
    else:
        report += "Relative layer depth has **negligible** impact on prediction.\n\n"

    if abs(inter_eff) > 0.01:
        report += "Feature × depth interactions provide **additional** predictive value beyond depth alone.\n\n"
    else:
        report += "Feature × depth interactions provide **no meaningful** additional value.\n\n"

    report += f"""---

## 5. Figures

### 5.1 Correlation Heatmap
![Correlation heatmap](./figures/01_correlation_heatmap.{FIGURE_FORMAT})

### 5.2 LASSO Coefficients
![LASSO coefficients](./figures/02_lasso_coefficients.{FIGURE_FORMAT})

### 5.3 {_TREE_ENGINE} Feature Importance
![{_TREE_ENGINE} feature importance](./figures/03_xgboost_importance.{FIGURE_FORMAT})

### 5.4 Predicted vs Oracle Scatter
![Predicted vs Oracle scatter](./figures/04_predicted_vs_oracle.{FIGURE_FORMAT})

### 5.5 Residual Plot
![Residual plot](./figures/05_residual_plot.{FIGURE_FORMAT})

### 5.6 Linearization Gap Summary
![Linearization Gap summary](./figures/06_linearization_gap.{FIGURE_FORMAT})

---

## 6. Scientific Conclusion

### Q1: Can the existing feature family explain Oracle Capability Drift?

{"**Yes.** The linear model achieves a Spearman ρ close to the nonlinear ceiling." if gap_abs < 0.05 else f"**Partially.** The linear model achieves ρ = {rho_lin:.4f}, but the tree model reaches ρ = {rho_tree:.4f}, leaving a gap of {gap:.4f}."}

### Q2: Does a nonlinear model significantly outperform a linear model?

{"**No.** The linearization gap is negligible." if gap_abs < 0.03 else f"**Yes.** The gap of {gap:.4f} indicates nonlinear structure beyond linear feature combinations." if gap_abs > 0.05 else f"**Moderately.** The gap of {gap:.4f} shows some unexploited nonlinear structure."}

### Q3: Is Experiment 2 scientifically justified?

{"**No.** The current feature family is sufficient." if gap_abs < 0.03 else "**Yes.** New features are needed to close the linearization gap." if gap_abs > 0.10 else f"**Conditionally.** The gap ({gap:.4f}) suggests room for improvement but is not extreme."}

---

## 7. Final Recommendation

{outcome_text}

---

*Report generated by CARE-MoE Experiment 1.5 pipeline.*
*Seed: {RANDOM_SEED} | Split: Strict Disjoint Expert (0–31 / 32–63)*
"""

    return report


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    set_global_seed()
    ensure_dirs()
    _set_pub_style()

    # ── Load artifacts ──
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    metrics_data = load_json(METRICS_PATH)
    all_metrics = metrics_data["performance"]
    gap_summary = metrics_data["linearization_gap"]
    y_test = np.array(metrics_data["y_test"])

    # Compute discarded count from original data
    from utils import load_raw_data
    from config import SEQ_LEN_FILTER
    raw_df = load_raw_data()
    raw_filtered = raw_df[raw_df["Seq_Len"] == SEQ_LEN_FILTER]
    n_discarded = len(raw_filtered) - len(train_df) - len(test_df)

    print(f"[Phase 3] Loaded train={len(train_df):,}  test={len(test_df):,}  discarded={n_discarded:,}")

    # ── Determine best models for scatter/residual plots ──
    best_overall = max(all_metrics, key=lambda m: m["Spearman"])
    best_key = f"{best_overall['Model']}_{best_overall['Variant']}"
    y_pred_best = np.array(metrics_data["predictions"][best_key])

    # Also find best LASSO variant for coefficient analysis
    lasso_metrics = [m for m in all_metrics if m["Model"] == "LASSO"]
    best_lasso = max(lasso_metrics, key=lambda m: m["Spearman"])
    best_lasso_key = f"LASSO_{best_lasso['Variant']}"

    # Load models
    lasso_model = load_pickle(os.path.join(MODELS_DIR, f"{best_lasso_key}.pkl"))

    # Find best XGBoost variant
    tree_name = [m["Model"] for m in all_metrics if m["Model"] not in {"LinearRegression", "Ridge", "LASSO"}][0]
    tree_metrics = [m for m in all_metrics if m["Model"] == tree_name]
    best_tree = max(tree_metrics, key=lambda m: m["Spearman"])
    best_tree_key = f"{best_tree['Model']}_{best_tree['Variant']}"
    xgb_model = load_pickle(os.path.join(MODELS_DIR, f"{best_tree_key}.pkl"))

    # Build feature name lists for the best variants
    _, col_names = build_feature_variants(test_df)
    lasso_features = col_names[best_lasso["Variant"]]
    tree_features = col_names[best_tree["Variant"]]

    # Build test arrays for SHAP
    test_variants, _ = build_feature_variants(test_df)
    X_test_tree = test_variants[best_tree["Variant"]]

    # ── Generate Figures ──
    figure_paths = {}

    print("\n[Phase 3] Generating publication-quality figures...")

    # 1. Correlation heatmap (on unscaled training data — reload raw)
    # For heatmap, use the raw-filtered training data to show actual correlations
    from config import TRAIN_EXPERTS
    raw_train = raw_filtered[
        raw_filtered["Expert_A"].isin(TRAIN_EXPERTS) &
        raw_filtered["Expert_B"].isin(TRAIN_EXPERTS)
    ]
    figure_paths["heatmap"] = plot_correlation_heatmap(raw_train)

    # 2. LASSO coefficients
    figure_paths["lasso"], lasso_coefs = plot_lasso_coefficients(
        lasso_model, lasso_features
    )

    # 3. XGBoost importance
    figure_paths["xgboost"] = plot_xgboost_importance(
        xgb_model, tree_features, X_test_tree
    )

    # 4. Predicted vs Oracle
    figure_paths["scatter"] = plot_predicted_vs_oracle(y_test, y_pred_best, best_key)

    # 5. Residuals
    figure_paths["residual"] = plot_residuals(y_test, y_pred_best, best_key)

    # 6. Linearization Gap
    figure_paths["gap"] = plot_linearization_gap(all_metrics, gap_summary)

    # ── Analysis ──
    lasso_analysis = analyze_lasso(lasso_model, lasso_features)
    variant_comparison = compare_variants(all_metrics)

    print(f"\n[Phase 3] LASSO active features : {len(lasso_analysis['active_features'])}")
    print(f"[Phase 3] LASSO dead features   : {len(lasso_analysis['dead_features'])}")
    print(f"[Phase 3] Depth effect (B−A)    : {variant_comparison['depth_effect_linear']:+.4f}")
    print(f"[Phase 3] Interaction effect (C−B): {variant_comparison['interaction_effect_linear']:+.4f}")

    # ── Generate Report ──
    report = generate_report(
        train_df=train_df,
        test_df=test_df,
        n_discarded=n_discarded,
        all_metrics=all_metrics,
        gap_summary=gap_summary,
        lasso_analysis=lasso_analysis,
        variant_comparison=variant_comparison,
        figure_paths=figure_paths,
    )

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n[Phase 3] Report → {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("PHASE 3 — REPRESENTATION ANALYSIS COMPLETE")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
