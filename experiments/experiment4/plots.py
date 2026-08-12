"""
CARE-MoE Experiment 4 — Plots
==============================
Six required plots per spec §23:

  1. Spearman rho by model (violin + points over 15 folds).
  2. RMSE by model.
  3. Δrho distribution (CA and BA).
  4. Partition-level Δrho (5 points explicitly labeled).
  5. Precision@K (K=10, 25, 50).
  6. Noise ceiling: SKIPPED panel with status message.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PLOTS_DIR,
    PARTITION_SEEDS,
    DELTA_RHO_MIN,
    PRECISION_K_VALUES,
    FIGURE_DPI,
    NOISE_CEILING_STATUS,
)


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


COLORS = {
    "A": "#2196F3",   # blue
    "B": "#FF9800",   # orange
    "C": "#4CAF50",   # green
}


def _save(fig, name: str) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 1: Spearman rho by model ─────────────────────────────────

def plot_spearman_by_model(fold_metrics: list) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))

    data_a = [m["rho_A"] for m in fold_metrics]
    data_b = [m["rho_B"] for m in fold_metrics]
    data_c = [m["rho_C"] for m in fold_metrics]

    positions = [1, 2, 3]
    vp = ax.violinplot([data_a, data_b, data_c], positions=positions,
                       showmedians=True, showextrema=True)
    for pc, col in zip(vp["bodies"], [COLORS["A"], COLORS["B"], COLORS["C"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part in vp:
            vp[part].set_color("black")
            vp[part].set_linewidth(1.2)

    # Jitter individual fold points
    rng = np.random.RandomState(42)
    for pos, data in zip(positions, [data_a, data_b, data_c]):
        jitter = rng.uniform(-0.08, 0.08, len(data))
        ax.scatter(np.full(len(data), pos) + jitter, data,
                   color="black", s=20, alpha=0.7, zorder=5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Model A\n(Local)", "Model B\n(Geometry)", "Model C\n(Local+Geo)"])
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Spearman ρ by Model — Middle Layer Only\n"
                 f"(15 folds: 5 partitions × 3 folds)", pad=10)

    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in ["A", "B", "C"]]
    ax.legend(handles=patches, loc="lower right")

    return _save(fig, "01_spearman_by_model")


# ── Plot 2: RMSE by model ──────────────────────────────────────────

def plot_rmse_by_model(fold_metrics: list) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))

    data_a = [m["rmse_A"] for m in fold_metrics]
    data_b = [m["rmse_B"] for m in fold_metrics]
    data_c = [m["rmse_C"] for m in fold_metrics]

    positions = [1, 2, 3]
    vp = ax.violinplot([data_a, data_b, data_c], positions=positions,
                       showmedians=True, showextrema=True)
    for pc, col in zip(vp["bodies"], [COLORS["A"], COLORS["B"], COLORS["C"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part in vp:
            vp[part].set_color("black")
            vp[part].set_linewidth(1.2)

    rng = np.random.RandomState(42)
    for pos, data in zip(positions, [data_a, data_b, data_c]):
        jitter = rng.uniform(-0.08, 0.08, len(data))
        ax.scatter(np.full(len(data), pos) + jitter, data,
                   color="black", s=20, alpha=0.7, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Model A\n(Local)", "Model B\n(Geometry)", "Model C\n(Local+Geo)"])
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Model — Middle Layer Only\n"
                 "(15 folds: 5 partitions × 3 folds)", pad=10)

    return _save(fig, "02_rmse_by_model")


# ── Plot 3: Δrho distribution (CA and BA) ─────────────────────────

def plot_delta_rho_distribution(fold_metrics: list) -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, key, title, color in [
        (axes[0], "delta_rho_CA", "Δρ_CA = ρ_C − ρ_A\n(CARE vs Local)", "#4CAF50"),
        (axes[1], "delta_rho_BA", "Δρ_BA = ρ_B − ρ_A\n(Geometry vs Local)", "#FF9800"),
    ]:
        vals = [m[key] for m in fold_metrics]
        ax.hist(vals, bins=8, color=color, alpha=0.7, edgecolor="black", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, label="0")
        ax.axvline(DELTA_RHO_MIN, color="red", linestyle=":", linewidth=1.2,
                   label=f"Δρ_min={DELTA_RHO_MIN}")
        ax.axvline(np.mean(vals), color=color, linestyle="-", linewidth=1.5,
                   label=f"mean={np.mean(vals):+.3f}")
        ax.set_xlabel("Δρ")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Δρ Distribution — 15 Folds (5 partitions × 3 folds)\n"
                 "Middle Layer Only", fontsize=11)
    plt.tight_layout()
    return _save(fig, "03_delta_rho_distribution")


# ── Plot 4: Partition-level Δrho (N=5 explicitly) ─────────────────

def plot_partition_delta_rho(partition_results: list, final_stats: dict) -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, key, title, color in [
        (axes[0], "delta_rho_CA_mean", "Δρ_CA per Partition\n(CARE vs Local)", "#4CAF50"),
        (axes[1], "delta_rho_BA_mean", "Δρ_BA per Partition\n(Geometry vs Local)", "#FF9800"),
    ]:
        x = list(range(1, 6))
        y = [p[key] for p in partition_results]
        seeds = [p.get("partition_seed", PARTITION_SEEDS[p["partition"]])
                 for p in partition_results]

        ax.scatter(x, y, color=color, s=80, zorder=5)
        ax.plot(x, y, color=color, alpha=0.5, linewidth=1)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(DELTA_RHO_MIN, color="red", linestyle=":", linewidth=1.2,
                   label=f"Δρ_min={DELTA_RHO_MIN}")

        mean_key = f"mean_{key}"
        lo_key = f"ci95_lo_{key}"
        hi_key = f"ci95_hi_{key}"
        if mean_key in final_stats:
            mean_v = final_stats[mean_key]
            lo_v = final_stats[lo_key]
            hi_v = final_stats[hi_key]
            ax.axhline(mean_v, color=color, linestyle="-", linewidth=1.5,
                       label=f"mean={mean_v:+.3f}")
            ax.fill_between([0.5, 5.5], lo_v, hi_v, alpha=0.15, color=color,
                            label=f"95% CI [{lo_v:+.3f}, {hi_v:+.3f}]")

        # Label each partition point
        for i, (xi, yi, seed) in enumerate(zip(x, y, seeds)):
            ax.annotate(f"P{i}\n(s={seed})", (xi, yi),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax.set_xlabel("Partition")
        ax.set_ylabel("Δρ")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        ax.text(0.02, 0.02, "N=5 partitions.\nCI width reflects low N.",
                transform=ax.transAxes, fontsize=7, color="gray",
                verticalalignment="bottom")

    fig.suptitle("Partition-Level Δρ — N=5 Independent Statistical Units\n"
                 "Middle Layer Only", fontsize=11)
    plt.tight_layout()
    return _save(fig, "04_partition_delta_rho")


# ── Plot 5: Precision@K ────────────────────────────────────────────

def plot_precision_at_k(fold_metrics: list) -> str:
    _style()
    fig, axes = plt.subplots(1, len(PRECISION_K_VALUES), figsize=(12, 4))
    if len(PRECISION_K_VALUES) == 1:
        axes = [axes]

    for ax, k_val in zip(axes, PRECISION_K_VALUES):
        means = {}
        for model in ["A", "B", "C"]:
            pk = f"prec_at_{k_val}_{model}"
            vals = [m.get(pk, float("nan")) for m in fold_metrics]
            means[model] = np.nanmean(vals)

        bars = ax.bar(["A", "B", "C"],
                      [means["A"], means["B"], means["C"]],
                      color=[COLORS["A"], COLORS["B"], COLORS["C"]],
                      alpha=0.8, edgecolor="black", linewidth=0.8)

        for bar, (model, val) in zip(bars, means.items()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Model")
        ax.set_ylabel("Precision@K")
        ax.set_title(f"Precision@{k_val}")
        ax.text(0.5, 0.95, f"K={k_val} (absolute)",
                transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    fig.suptitle("Precision@K — Mean over 15 Folds\n"
                 "Safe = lowest Oracle KL pairs. Middle Layer Only.", fontsize=11)
    plt.tight_layout()
    return _save(fig, "05_precision_at_k")


# ── Plot 6: Noise ceiling status ──────────────────────────────────

def plot_noise_ceiling() -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    text = (
        f"Noise Ceiling Status: {NOISE_CEILING_STATUS}\n\n"
        "Reason: No genuine repeated Oracle measurements are available.\n"
        "Different Seq_Len values (64, 128, 256, 512) are NOT independent\n"
        "replicates of the same measurement.\n\n"
        "To activate: provide independent repeated Oracle measurements\n"
        "for ~100 pairs, stratified across the Oracle distance distribution."
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#FF9800"),
            fontfamily="monospace")
    ax.set_title("Noise Ceiling — Experiment 4", fontsize=12)

    return _save(fig, "06_noise_ceiling_status")


# ── Main ──────────────────────────────────────────────────────────

def generate_all_plots(
    fold_metrics: list,
    partition_results: list,
    final_stats: dict,
) -> list:
    """Generate all 6 required plots. Returns list of saved paths."""
    paths = []
    paths.append(plot_spearman_by_model(fold_metrics))
    paths.append(plot_rmse_by_model(fold_metrics))
    paths.append(plot_delta_rho_distribution(fold_metrics))
    paths.append(plot_partition_delta_rho(partition_results, final_stats))
    paths.append(plot_precision_at_k(fold_metrics))
    paths.append(plot_noise_ceiling())
    for p in paths:
        print(f"  [plots] → {p}")
    return paths


if __name__ == "__main__":
    print("Run plots via run_all.py Phase 6.")
