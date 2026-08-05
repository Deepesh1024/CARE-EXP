"""
CARE-MoE Experiment 2 — Phase 6: Linearization Gap Comparison
================================================================
Compare Experiment 1.5 baseline vs Experiment 2 augmented results.

Computes:
  - Gap reduction ΔΔ
  - Statistical significance via bootstrap
  - Within-layer Spearman ρ analysis
  - Hypothesis outcome determination

Produces:
  results/exp2/plots/regression/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import (
    EXP15_METRICS_PATH,
    METRICS_PATH,
    TARGET,
    TEST_PARQUET,
    RANDOM_SEED,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_json,
    save_json,
    set_pub_style,
    save_fig,
)


def load_both_gaps():
    """Load linearization gap from both experiments."""
    exp15 = load_json(EXP15_METRICS_PATH)
    exp2 = load_json(METRICS_PATH)

    gap_15 = exp15["linearization_gap"]
    gap_2 = exp2["linearization_gap"]

    print("--- Experiment 1.5 Baseline ---")
    print(f"  Best Linear: {gap_15['best_linear_model']} "
          f"(ρ = {gap_15['best_linear_spearman']:.4f})")
    print(f"  Best Tree:   {gap_15['best_tree_model']} "
          f"(ρ = {gap_15['best_tree_spearman']:.4f})")
    print(f"  Δ_gap:       {gap_15['linearization_gap']:.4f}")

    print("\n--- Experiment 2 (Augmented) ---")
    print(f"  Best Linear: {gap_2['best_linear_model']} "
          f"(ρ = {gap_2['best_linear_spearman']:.4f})")
    print(f"  Best Tree:   {gap_2['best_tree_model']} "
          f"(ρ = {gap_2['best_tree_spearman']:.4f})")
    print(f"  Δ_gap:       {gap_2['linearization_gap']:.4f}")

    delta_gap = gap_15["linearization_gap"] - gap_2["linearization_gap"]
    print(f"\n  ΔΔ (gap reduction): {delta_gap:+.4f}")

    return gap_15, gap_2, delta_gap


def bootstrap_gap_significance(n_boot: int = 1000):
    """Bootstrap test for statistical significance of gap reduction."""
    rng = np.random.RandomState(RANDOM_SEED)

    exp15 = load_json(EXP15_METRICS_PATH)
    exp2 = load_json(METRICS_PATH)

    y_test = np.array(exp2["y_test"])

    # Best models from each experiment
    pred_15_linear = np.array(exp15["predictions"]["LinearRegression_A"])
    pred_15_tree = np.array(exp15["predictions"]["XGBoost_B"])
    pred_2_linear = np.array(exp2["predictions"][exp2["linearization_gap"]["best_linear_model"]])
    pred_2_tree = np.array(exp2["predictions"][exp2["linearization_gap"]["best_tree_model"]])

    n = len(y_test)
    gaps_15 = []
    gaps_2 = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_b = y_test[idx]

        sp_15_l, _ = spearmanr(y_b, pred_15_linear[idx])
        sp_15_t, _ = spearmanr(y_b, pred_15_tree[idx])
        sp_2_l, _ = spearmanr(y_b, pred_2_linear[idx])
        sp_2_t, _ = spearmanr(y_b, pred_2_tree[idx])

        gaps_15.append(sp_15_t - sp_15_l)
        gaps_2.append(sp_2_t - sp_2_l)

    gaps_15 = np.array(gaps_15)
    gaps_2 = np.array(gaps_2)
    delta_gaps = gaps_15 - gaps_2

    p_value = np.mean(delta_gaps <= 0)  # fraction where gap didn't reduce

    print(f"\n--- Bootstrap ({n_boot} iterations) ---")
    print(f"  Exp 1.5 gap: {gaps_15.mean():.4f} ± {gaps_15.std():.4f}")
    print(f"  Exp 2 gap:   {gaps_2.mean():.4f} ± {gaps_2.std():.4f}")
    print(f"  ΔΔ mean:     {delta_gaps.mean():+.4f}")
    print(f"  p-value:     {p_value:.4f}")

    return {
        "gap_15_mean": float(gaps_15.mean()),
        "gap_15_std": float(gaps_15.std()),
        "gap_2_mean": float(gaps_2.mean()),
        "gap_2_std": float(gaps_2.std()),
        "delta_mean": float(delta_gaps.mean()),
        "delta_std": float(delta_gaps.std()),
        "p_value": float(p_value),
    }


def within_layer_analysis():
    """Compute Spearman ρ separately for each layer."""
    test_df = pd.read_parquet(TEST_PARQUET)
    y_test = test_df[TARGET].values

    exp2 = load_json(METRICS_PATH)
    best_linear = exp2["linearization_gap"]["best_linear_model"]
    best_tree = exp2["linearization_gap"]["best_tree_model"]

    pred_linear = np.array(exp2["predictions"][best_linear])
    pred_tree = np.array(exp2["predictions"][best_tree])

    print("\n--- Within-Layer Spearman ρ (Experiment 2) ---")
    rows = []
    for layer in sorted(test_df["Layer"].unique()):
        mask = test_df["Layer"] == layer
        y_l = y_test[mask.values]
        p_lin = pred_linear[mask.values]
        p_tree = pred_tree[mask.values]

        sp_lin, _ = spearmanr(y_l, p_lin)
        sp_tree, _ = spearmanr(y_l, p_tree)

        rows.append({
            "Layer": layer,
            "N": int(mask.sum()),
            "Linear_rho": float(sp_lin),
            "Tree_rho": float(sp_tree),
            "Gap": float(sp_tree - sp_lin),
        })
        print(f"  {layer:8s}: Linear ρ={sp_lin:+.4f}, "
              f"Tree ρ={sp_tree:+.4f}, gap={sp_tree-sp_lin:+.4f}")

    return pd.DataFrame(rows)


def plot_gap_comparison(gap_15, gap_2):
    """Side-by-side bar chart comparing gaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Spearman ρ comparison
    ax = axes[0]
    experiments = ["Exp 1.5\n(Baseline)", "Exp 2\n(Augmented)"]
    linear_rhos = [gap_15["best_linear_spearman"],
                    gap_2["best_linear_spearman"]]
    tree_rhos = [gap_15["best_tree_spearman"],
                  gap_2["best_tree_spearman"]]

    x = np.arange(len(experiments))
    width = 0.3
    bars1 = ax.bar(x - width/2, linear_rhos, width, label="Best Linear",
                    color="#42A5F5", edgecolor="white")
    bars2 = ax.bar(x + width/2, tree_rhos, width, label="Best Tree",
                    color="#EF5350", edgecolor="white")

    ax.set_ylabel("Spearman ρ")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.set_ylim(0, max(max(linear_rhos), max(tree_rhos)) * 1.15)

    # Annotate values
    for bar, val in zip(bars1, linear_rhos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, tree_rhos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Linearization Gap
    ax = axes[1]
    gaps = [gap_15["linearization_gap"], gap_2["linearization_gap"]]
    colors = ["#FF7043", "#66BB6A"]
    bars = ax.bar(experiments, gaps, color=colors, edgecolor="white", width=0.5)

    ax.set_ylabel("Linearization Gap (Δ)")
    ax.set_title("Linearization Gap Comparison")

    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"Δ = {val:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    fig.suptitle("Experiment 1.5 vs. Experiment 2: Linearization Gap Analysis",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "gap_comparison", subdir="regression")


def plot_predicted_vs_actual():
    """Predicted vs actual scatter for best Exp 2 models."""
    exp2 = load_json(METRICS_PATH)
    y_test = np.array(exp2["y_test"])
    best_linear = exp2["linearization_gap"]["best_linear_model"]
    best_tree = exp2["linearization_gap"]["best_tree_model"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model_key, title, color in [
        (axes[0], best_linear, f"Best Linear ({best_linear})", "#1976D2"),
        (axes[1], best_tree, f"Best Tree ({best_tree})", "#E64A19"),
    ]:
        pred = np.array(exp2["predictions"][model_key])
        ax.scatter(y_test, pred, alpha=0.3, s=10, color=color)

        # Perfect prediction line
        lims = [min(y_test.min(), pred.min()),
                max(y_test.max(), pred.max())]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

        sp, _ = spearmanr(y_test, pred)
        ax.set_xlabel("Actual Oracle KL")
        ax.set_ylabel("Predicted Oracle KL")
        ax.set_title(f"{title}\nρ = {sp:+.4f}")

    fig.suptitle("Predicted vs. Actual (Experiment 2)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "predicted_vs_actual", subdir="regression")


def determine_hypothesis_outcome(gap_15, gap_2, bootstrap):
    """Determine and print hypothesis test outcome."""
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST OUTCOME")
    print("=" * 60)

    delta_linear = gap_2["best_linear_spearman"] - gap_15["best_linear_spearman"]
    delta_tree = gap_2["best_tree_spearman"] - gap_15["best_tree_spearman"]
    gap_reduction = gap_15["linearization_gap"] - gap_2["linearization_gap"]

    print(f"  Linear ρ improvement: {delta_linear:+.4f}")
    print(f"  Tree ρ improvement:   {delta_tree:+.4f}")
    print(f"  Gap reduction:        {gap_reduction:+.4f}")
    print(f"  Bootstrap p-value:    {bootstrap['p_value']:.4f}")

    if gap_reduction > 0 and bootstrap["p_value"] < 0.05:
        verdict = ("REJECT H₀: The new capability descriptors significantly "
                    "reduce the Linearization Gap.")
    elif gap_reduction > 0:
        verdict = ("PARTIAL SUPPORT: Gap reduced but not statistically "
                    "significant (p > 0.05).")
    else:
        verdict = ("FAIL TO REJECT H₀: The new descriptors did not reduce "
                    "the Linearization Gap.")

    print(f"\n  VERDICT: {verdict}")
    print("=" * 60)

    return {
        "delta_linear_rho": float(delta_linear),
        "delta_tree_rho": float(delta_tree),
        "gap_reduction": float(gap_reduction),
        "p_value": bootstrap["p_value"],
        "verdict": verdict,
    }


def main():
    set_global_seed()
    ensure_dirs()
    set_pub_style()

    print("=" * 70)
    print("PHASE 6 — LINEARIZATION GAP COMPARISON")
    print("=" * 70)

    # Load and compare gaps
    gap_15, gap_2, delta_gap = load_both_gaps()

    # Bootstrap significance
    bootstrap = bootstrap_gap_significance(1000)

    # Within-layer analysis
    within_df = within_layer_analysis()

    # Plots
    plot_gap_comparison(gap_15, gap_2)
    plot_predicted_vs_actual()

    # Hypothesis outcome
    outcome = determine_hypothesis_outcome(gap_15, gap_2, bootstrap)

    # Update metrics.json with gap comparison
    metrics = load_json(METRICS_PATH)
    metrics["gap_comparison"] = {
        "exp15_gap": gap_15,
        "exp2_gap": gap_2,
        "bootstrap": bootstrap,
        "within_layer": within_df.to_dict(orient="records"),
        "hypothesis_outcome": outcome,
    }
    save_json(metrics, METRICS_PATH)

    print("\n[Phase 6] Linearization Gap Comparison complete.")
    return gap_15, gap_2, bootstrap, outcome


if __name__ == "__main__":
    main()
