"""
CARE-MoE Experiment 3B — Phase 4: Statistical Analysis & Figures
==================================================================
1. Aggregate CV results across folds/repetitions/realizations.
2. Compute mean, std, 95% CI for all metrics per q.
3. Oracle vs Null A and Oracle vs Null B statistical comparisons.
4. Generate Primary Figures 1 & 2 and summary tables.
5. Save dimension_summary.csv and dimension_summary.json.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR,
    FIGURES_DIR,
    LAYERS,
    Q_VALUES,
    RANDOM_SEED,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_json,
    save_csv,
    set_pub_style,
    save_fig,
)


def compute_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Compute mean and 95% confidence interval.

    Returns (mean, ci_low, ci_high).
    """
    values = values[np.isfinite(values)]
    if len(values) == 1:
        return (float(values[0]), float("nan"), float("nan"))
    elif len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    ci_half = 1.96 * se
    return (float(mean), float(mean - ci_half), float(mean + ci_half))


def aggregate_results(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Aggregate CV results to produce mean ± 95% CI per group."""
    rows = []
    for group_keys, group_df in df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(group_cols, group_keys))
        for metric in metric_cols:
            vals = group_df[metric].values
            mean, ci_lo, ci_hi = compute_ci95(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = ci_lo
            row[f"{metric}_ci_hi"] = ci_hi
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1))
            row[f"{metric}_n"] = int(np.sum(np.isfinite(vals)))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_null_results(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Aggregate null CV results: first average within each realization, then across.

    This avoids pseudo-replication from multiple folds within one realization.
    """
    # First: average metrics per (layer, q, realization) across folds/reps
    realization_means = df.groupby(["layer", "q", "realization"])[metric_cols].mean().reset_index()

    # Then: aggregate across realizations
    rows = []
    for (layer, q), group_df in realization_means.groupby(["layer", "q"]):
        row = {"layer": layer, "q": q}
        for metric in metric_cols:
            vals = group_df[metric].values
            mean, ci_lo, ci_hi = compute_ci95(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = ci_lo
            row[f"{metric}_ci_hi"] = ci_hi
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1))
            row[f"{metric}_n"] = int(np.sum(np.isfinite(vals)))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fidelity_curves(oracle_agg, null_a_agg, null_b_agg, layer: str):
    """PRIMARY FIGURE 1: q vs held-out test→test Spearman ρ."""
    set_pub_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    metric = "test_test_spearman"

    # Oracle
    oracle_layer = oracle_agg[oracle_agg["layer"] == layer].sort_values("q")
    ax.plot(oracle_layer["q"], oracle_layer[f"{metric}_mean"],
            "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="Oracle", zorder=5)
    ax.fill_between(oracle_layer["q"],
                     oracle_layer[f"{metric}_ci_lo"],
                     oracle_layer[f"{metric}_ci_hi"],
                     alpha=0.2, color="#1f77b4", zorder=4)

    # Null A
    na_layer = null_a_agg[null_a_agg["layer"] == layer].sort_values("q")
    ax.plot(na_layer["q"], na_layer[f"{metric}_mean"],
            "s--", color="#ff7f0e", linewidth=2, markersize=6, label="Null A (Shuffled)", zorder=3)
    ax.fill_between(na_layer["q"],
                     na_layer[f"{metric}_ci_lo"],
                     na_layer[f"{metric}_ci_hi"],
                     alpha=0.15, color="#ff7f0e", zorder=2)

    # Null B
    nb_layer = null_b_agg[null_b_agg["layer"] == layer].sort_values("q")
    ax.plot(nb_layer["q"], nb_layer[f"{metric}_mean"],
            "^--", color="#2ca02c", linewidth=2, markersize=6, label="Null B (Random Euclidean)", zorder=3)
    ax.fill_between(nb_layer["q"],
                     nb_layer[f"{metric}_ci_lo"],
                     nb_layer[f"{metric}_ci_hi"],
                     alpha=0.15, color="#2ca02c", zorder=2)

    ax.set_xlabel("Embedding Dimension q")
    ax.set_ylabel("Held-Out Test→Test Spearman ρ")
    ax.set_title(f"Capability Geometry Fidelity — Layer: {layer}")
    ax.set_xticks(Q_VALUES)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return fig


def plot_stress_curves(oracle_agg, null_a_agg, null_b_agg, layer: str):
    """PRIMARY FIGURE 2: q vs held-out test→test RMSE."""
    set_pub_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    metric = "test_test_rmse"

    # Oracle
    oracle_layer = oracle_agg[oracle_agg["layer"] == layer].sort_values("q")
    ax.plot(oracle_layer["q"], oracle_layer[f"{metric}_mean"],
            "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="Oracle", zorder=5)
    ax.fill_between(oracle_layer["q"],
                     oracle_layer[f"{metric}_ci_lo"],
                     oracle_layer[f"{metric}_ci_hi"],
                     alpha=0.2, color="#1f77b4", zorder=4)

    # Null A
    na_layer = null_a_agg[null_a_agg["layer"] == layer].sort_values("q")
    ax.plot(na_layer["q"], na_layer[f"{metric}_mean"],
            "s--", color="#ff7f0e", linewidth=2, markersize=6, label="Null A (Shuffled)", zorder=3)
    ax.fill_between(na_layer["q"],
                     na_layer[f"{metric}_ci_lo"],
                     na_layer[f"{metric}_ci_hi"],
                     alpha=0.15, color="#ff7f0e", zorder=2)

    # Null B
    nb_layer = null_b_agg[null_b_agg["layer"] == layer].sort_values("q")
    ax.plot(nb_layer["q"], nb_layer[f"{metric}_mean"],
            "^--", color="#2ca02c", linewidth=2, markersize=6, label="Null B (Random Euclidean)", zorder=3)
    ax.fill_between(nb_layer["q"],
                     nb_layer[f"{metric}_ci_lo"],
                     nb_layer[f"{metric}_ci_hi"],
                     alpha=0.15, color="#2ca02c", zorder=2)

    ax.set_xlabel("Embedding Dimension q")
    ax.set_ylabel("Held-Out Test→Test RMSE")
    ax.set_title(f"Embedding Error — Layer: {layer}")
    ax.set_xticks(Q_VALUES)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return fig


def plot_combined_fidelity(oracle_agg, null_a_agg, null_b_agg):
    """Combined fidelity curve across all layers (3-panel figure)."""
    set_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    metric = "test_test_spearman"

    for idx, layer in enumerate(LAYERS):
        ax = axes[idx]

        # Oracle
        ol = oracle_agg[oracle_agg["layer"] == layer].sort_values("q")
        ax.plot(ol["q"], ol[f"{metric}_mean"],
                "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="Oracle", zorder=5)
        ax.fill_between(ol["q"], ol[f"{metric}_ci_lo"], ol[f"{metric}_ci_hi"],
                         alpha=0.2, color="#1f77b4", zorder=4)

        # Null A
        na = null_a_agg[null_a_agg["layer"] == layer].sort_values("q")
        ax.plot(na["q"], na[f"{metric}_mean"],
                "s--", color="#ff7f0e", linewidth=2, markersize=6, label="Null A (Shuffled)", zorder=3)
        ax.fill_between(na["q"], na[f"{metric}_ci_lo"], na[f"{metric}_ci_hi"],
                         alpha=0.15, color="#ff7f0e", zorder=2)

        # Null B
        nb = null_b_agg[null_b_agg["layer"] == layer].sort_values("q")
        ax.plot(nb["q"], nb[f"{metric}_mean"],
                "^--", color="#2ca02c", linewidth=2, markersize=6, label="Null B (Random Eucl.)", zorder=3)
        ax.fill_between(nb["q"], nb[f"{metric}_ci_lo"], nb[f"{metric}_ci_hi"],
                         alpha=0.15, color="#2ca02c", zorder=2)

        ax.set_xlabel("Embedding Dimension q")
        ax.set_title(f"Layer: {layer}")
        ax.set_xticks(Q_VALUES)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Held-Out Test→Test Spearman ρ")
        if idx == 2:
            ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Capability Geometry Fidelity — Expert-Level Holdout", fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def build_summary_table(oracle_agg, null_a_agg, null_b_agg, layer: str) -> pd.DataFrame:
    """Build the required summary table per layer."""
    rows = []
    for q in Q_VALUES:
        o = oracle_agg[(oracle_agg["layer"] == layer) & (oracle_agg["q"] == q)].iloc[0]
        a = null_a_agg[(null_a_agg["layer"] == layer) & (null_a_agg["q"] == q)].iloc[0]
        b = null_b_agg[(null_b_agg["layer"] == layer) & (null_b_agg["q"] == q)].iloc[0]

        def fmt(mean_key, ci_lo_key, ci_hi_key, src):
            m = src[mean_key]
            lo = src[ci_lo_key]
            hi = src[ci_hi_key]
            return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

        rows.append({
            "q": q,
            "Oracle_rho": fmt("test_test_spearman_mean", "test_test_spearman_ci_lo", "test_test_spearman_ci_hi", o),
            "NullA_rho": fmt("test_test_spearman_mean", "test_test_spearman_ci_lo", "test_test_spearman_ci_hi", a),
            "NullB_rho": fmt("test_test_spearman_mean", "test_test_spearman_ci_lo", "test_test_spearman_ci_hi", b),
            "Oracle_RMSE": fmt("test_test_rmse_mean", "test_test_rmse_ci_lo", "test_test_rmse_ci_hi", o),
            "NullA_RMSE": fmt("test_test_rmse_mean", "test_test_rmse_ci_lo", "test_test_rmse_ci_hi", a),
            "NullB_RMSE": fmt("test_test_rmse_mean", "test_test_rmse_ci_lo", "test_test_rmse_ci_hi", b),
            "Oracle_Stress": fmt("test_test_norm_stress_mean", "test_test_norm_stress_ci_lo", "test_test_norm_stress_ci_hi", o),
            "NullA_Stress": fmt("test_test_norm_stress_mean", "test_test_norm_stress_ci_lo", "test_test_norm_stress_ci_hi", a),
            "NullB_Stress": fmt("test_test_norm_stress_mean", "test_test_norm_stress_ci_lo", "test_test_norm_stress_ci_hi", b),
        })
    return pd.DataFrame(rows)


def compute_statistical_comparisons(oracle_agg, null_a_agg, null_b_agg) -> list[dict]:
    """Compare Oracle vs Null A and Oracle vs Null B for each (layer, q)."""
    comparisons = []
    for layer in LAYERS:
        for q in Q_VALUES:
            o = oracle_agg[(oracle_agg["layer"] == layer) & (oracle_agg["q"] == q)].iloc[0]
            a = null_a_agg[(null_a_agg["layer"] == layer) & (null_a_agg["q"] == q)].iloc[0]
            b = null_b_agg[(null_b_agg["layer"] == layer) & (null_b_agg["q"] == q)].iloc[0]

            metric = "test_test_spearman"

            # Oracle - Null A
            diff_a = o[f"{metric}_mean"] - a[f"{metric}_mean"]
            # Conservative CI for difference using sum of half-widths
            o_hw = (o[f"{metric}_ci_hi"] - o[f"{metric}_ci_lo"]) / 2
            a_hw = (a[f"{metric}_ci_hi"] - a[f"{metric}_ci_lo"]) / 2
            diff_a_hw = np.sqrt(o_hw**2 + a_hw**2)

            # Oracle - Null B
            diff_b = o[f"{metric}_mean"] - b[f"{metric}_mean"]
            b_hw = (b[f"{metric}_ci_hi"] - b[f"{metric}_ci_lo"]) / 2
            diff_b_hw = np.sqrt(o_hw**2 + b_hw**2)

            comparisons.append({
                "layer": layer,
                "q": q,
                "oracle_minus_null_a_rho": float(diff_a),
                "oracle_minus_null_a_ci_lo": float(diff_a - 1.96 * diff_a_hw / 1.96),  # propagated
                "oracle_minus_null_a_ci_hi": float(diff_a + 1.96 * diff_a_hw / 1.96),
                "oracle_minus_null_b_rho": float(diff_b),
                "oracle_minus_null_b_ci_lo": float(diff_b - diff_b_hw),
                "oracle_minus_null_b_ci_hi": float(diff_b + diff_b_hw),
                "ci_excludes_zero_vs_null_a": bool(diff_a - diff_a_hw > 0),
                "ci_excludes_zero_vs_null_b": bool(diff_b - diff_b_hw > 0),
            })
    return comparisons


def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 3B — PHASE 4: STATISTICAL ANALYSIS & FIGURES")
    print("=" * 70)

    # ── Load CV Results ───────────────────────────
    oracle_df = pd.read_csv(os.path.join(RESULTS_DIR, "oracle_cv_results.csv"))
    null_a_df = pd.read_csv(os.path.join(RESULTS_DIR, "null_a_cv_results.csv"))
    null_b_df = pd.read_csv(os.path.join(RESULTS_DIR, "null_b_cv_results.csv"))

    print(f"[Phase 4] Loaded Oracle CV: {len(oracle_df)} rows")
    print(f"[Phase 4] Loaded Null A CV: {len(null_a_df)} rows")
    print(f"[Phase 4] Loaded Null B CV: {len(null_b_df)} rows")

    metric_cols = [
        "test_train_spearman", "test_train_pearson",
        "test_train_rmse", "test_train_mae", "test_train_norm_stress",
        "test_test_spearman", "test_test_pearson",
        "test_test_rmse", "test_test_mae", "test_test_norm_stress",
    ]

    # ── Aggregate ─────────────────────────────────
    print("\n[Phase 4] Aggregating Oracle results...")
    oracle_agg = aggregate_results(oracle_df, ["layer", "q"], metric_cols)

    print("[Phase 4] Aggregating Null A results (realization-level)...")
    null_a_agg = aggregate_null_results(null_a_df, metric_cols)

    print("[Phase 4] Aggregating Null B results (realization-level)...")
    null_b_agg = aggregate_null_results(null_b_df, metric_cols)

    # ── Generate Figures ──────────────────────────
    print("\n[Phase 4] Generating figures...")

    # Per-layer figures
    for layer in LAYERS:
        fig1 = plot_fidelity_curves(oracle_agg, null_a_agg, null_b_agg, layer)
        save_fig(fig1, f"fidelity_curve_{layer}")

        fig2 = plot_stress_curves(oracle_agg, null_a_agg, null_b_agg, layer)
        save_fig(fig2, f"stress_curve_{layer}")

    # Combined multi-panel figure
    fig_combined = plot_combined_fidelity(oracle_agg, null_a_agg, null_b_agg)
    save_fig(fig_combined, "fidelity_curve")

    # ── Summary Tables ────────────────────────────
    print("\n[Phase 4] Building summary tables...")
    all_summary_rows = []
    for layer in LAYERS:
        table = build_summary_table(oracle_agg, null_a_agg, null_b_agg, layer)
        table.insert(0, "layer", layer)
        all_summary_rows.append(table)
        print(f"\n  Summary for layer: {layer}")
        print(table.to_string(index=False))

    summary_df = pd.concat(all_summary_rows, ignore_index=True)
    save_csv(summary_df, os.path.join(RESULTS_DIR, "dimension_summary.csv"))

    # ── Numerical Data Behind Curves ──────────────
    save_csv(oracle_agg, os.path.join(RESULTS_DIR, "oracle_agg_stats.csv"))
    save_csv(null_a_agg, os.path.join(RESULTS_DIR, "null_a_agg_stats.csv"))
    save_csv(null_b_agg, os.path.join(RESULTS_DIR, "null_b_agg_stats.csv"))

    # ── Statistical Comparisons ───────────────────
    print("\n[Phase 4] Computing statistical comparisons...")
    comparisons = compute_statistical_comparisons(oracle_agg, null_a_agg, null_b_agg)
    comp_df = pd.DataFrame(comparisons)
    save_csv(comp_df, os.path.join(RESULTS_DIR, "statistical_comparisons.csv"))

    print("\n  Oracle vs Null comparisons:")
    for _, row in comp_df.iterrows():
        print(f"  Layer={row['layer']}, q={row['q']:2d}: "
              f"Δρ(A)={row['oracle_minus_null_a_rho']:+.4f} "
              f"[CI excl 0: {row['ci_excludes_zero_vs_null_a']}], "
              f"Δρ(B)={row['oracle_minus_null_b_rho']:+.4f} "
              f"[CI excl 0: {row['ci_excludes_zero_vs_null_b']}]")

    # ── Dimension Summary JSON ────────────────────
    dimension_summary = {
        "layers_analyzed": LAYERS,
        "q_values": Q_VALUES,
        "per_layer": {},
    }
    for layer in LAYERS:
        layer_summary = {}
        for q in Q_VALUES:
            o = oracle_agg[(oracle_agg["layer"] == layer) & (oracle_agg["q"] == q)].iloc[0]
            a = null_a_agg[(null_a_agg["layer"] == layer) & (null_a_agg["q"] == q)].iloc[0]
            b = null_b_agg[(null_b_agg["layer"] == layer) & (null_b_agg["q"] == q)].iloc[0]
            comp = comp_df[(comp_df["layer"] == layer) & (comp_df["q"] == q)].iloc[0]

            layer_summary[str(q)] = {
                "oracle_test_test_spearman": {
                    "mean": o["test_test_spearman_mean"],
                    "ci_lo": o["test_test_spearman_ci_lo"],
                    "ci_hi": o["test_test_spearman_ci_hi"],
                },
                "null_a_test_test_spearman": {
                    "mean": a["test_test_spearman_mean"],
                    "ci_lo": a["test_test_spearman_ci_lo"],
                    "ci_hi": a["test_test_spearman_ci_hi"],
                },
                "null_b_test_test_spearman": {
                    "mean": b["test_test_spearman_mean"],
                    "ci_lo": b["test_test_spearman_ci_lo"],
                    "ci_hi": b["test_test_spearman_ci_hi"],
                },
                "oracle_minus_null_a": float(comp["oracle_minus_null_a_rho"]),
                "oracle_minus_null_b": float(comp["oracle_minus_null_b_rho"]),
                "ci_excludes_zero_vs_null_a": bool(comp["ci_excludes_zero_vs_null_a"]),
                "ci_excludes_zero_vs_null_b": bool(comp["ci_excludes_zero_vs_null_b"]),
            }
        dimension_summary["per_layer"][layer] = layer_summary

    save_json(dimension_summary, os.path.join(RESULTS_DIR, "dimension_summary.json"))

    print("\n" + "=" * 70)
    print("PHASE 4 — STATISTICAL ANALYSIS & FIGURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
