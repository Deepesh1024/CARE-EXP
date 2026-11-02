#!/usr/bin/env python3
"""
Generate publication-quality plots for the full multi-seed OLMoE-1B-7B benchmark matrix on NVIDIA L40S.
Includes:
1. 4-Panel Master Benchmark Summary (PPL vs Experts, Causal KL vs Experts, 48E Breakdown, Time-PPL Pareto)
2. Standalone Figures for PPL, KL, 48E Bar Comparison, and Pareto Trade-off.
"""

import os
import glob
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from collections import defaultdict

# ---------------------------------------------------------
# Style & Configuration
# ---------------------------------------------------------
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "font.family": "sans-serif",
    "figure.autolayout": False,
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.0,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ARTIFACT_DIR = "/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b"

RUN_DIRS = [
    os.path.join(BASE_DIR, "run_20261023T105248Z_c4da7cdb"),
    os.path.join(BASE_DIR, "run_20261025T190236Z_fd94e9fb"),
]

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
all_runs = []
for r_dir in RUN_DIRS:
    for filepath in glob.glob(os.path.join(r_dir, "*_result.json")):
        with open(filepath, "r") as fp:
            all_runs.append(json.load(fp))

print(f"Loaded {len(all_runs)} runs.")

# Map data by method and target
# method -> target -> list of run dicts
data = defaultdict(lambda: defaultdict(list))
for run in all_runs:
    data[run["method"]][run["target_experts"]].append(run)

# Method Display Metadata
# (Label, Paradigm, Color, Marker, Linestyle, Z-order)
METHOD_META = {
    "care_adaptive": {
        "label": "CARE-Adaptive (Ours)",
        "paradigm": "Merging",
        "color": "#1565C0",      # Deep Royal Blue
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 8,
        "zorder": 10,
    },
    "care_static": {
        "label": "CARE-Static",
        "paradigm": "Merging",
        "color": "#42A5F5",      # Sky Blue
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.0,
        "markersize": 7,
        "zorder": 8,
    },
    "mc_smoe": {
        "label": "MC-SMoE (Stages 1-4)",
        "paradigm": "Merging",
        "color": "#FF7043",      # Vibrant Coral
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 2.0,
        "markersize": 7,
        "zorder": 7,
    },
    "hc_smoe": {
        "label": "HC-SMoE (Router-proxy)",
        "paradigm": "Merging",
        "color": "#8E24AA",      # Deep Purple
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.0,
        "markersize": 7,
        "zorder": 6,
    },
    "sub_moe": {
        "label": "Sub-MoE (Joint SVD)",
        "paradigm": "Merging",
        "color": "#D32F2F",      # Crimson Red
        "marker": "v",
        "linestyle": ":",
        "linewidth": 2.2,
        "markersize": 7,
        "zorder": 5,
    },
    "random": {
        "label": "Random-K5 (Control)",
        "paradigm": "Control",
        "color": "#78909C",      # Slate Gray
        "marker": "p",
        "linestyle": ":",
        "linewidth": 1.8,
        "markersize": 7,
        "zorder": 4,
    },
    "reap_prune": {
        "label": "REAP-Prune (Pruning)",
        "paradigm": "Pruning",
        "color": "#2E7D32",      # Forest Green
        "marker": "X",
        "linestyle": "--",
        "linewidth": 2.2,
        "markersize": 8,
        "zorder": 9,
    },
}

TARGETS = [60, 56, 48]
BASE_PPL = 10.2804

# Compute statistics with sample standard deviation (ddof=1)
stats = {}
for m, targets_dict in data.items():
    stats[m] = {}
    for t in TARGETS:
        runs = targets_dict[t]
        ppls = [r["compressed_ppl"] for r in runs]
        kls = [r["heldout_test_kl"] for r in runs]
        runtimes = [r["compression_time_s"] for r in runs]
        stats[m][t] = {
            "ppl_mean": float(np.mean(ppls)),
            "ppl_std": float(np.std(ppls, ddof=1)),
            "ppl_vals": ppls,
            "kl_mean": float(np.mean(kls)),
            "kl_std": float(np.std(kls, ddof=1)),
            "kl_vals": kls,
            "time_mean": float(np.mean(runtimes)),
            "time_std": float(np.std(runtimes, ddof=1)),
        }

# ---------------------------------------------------------
# 1. Flagship 4-Panel Master Benchmark Summary Figure
# ---------------------------------------------------------
def create_4panel_master_figure():
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    
    # ---------------- Panel A: PPL vs Active Experts ----------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Reference baseline line
    ax_a.axhline(BASE_PPL, color="#424242", linestyle=":", linewidth=1.5, zorder=2,
                 label=f"64E Uncompressed Base ({BASE_PPL:.2f})")
    
    # Plot curves
    plot_methods_order = ["reap_prune", "care_adaptive", "mc_smoe", "care_static", "random", "hc_smoe", "sub_moe"]
    for m in plot_methods_order:
        meta = METHOD_META[m]
        x_pts = TARGETS
        y_pts = [stats[m][t]["ppl_mean"] for t in TARGETS]
        y_err = [stats[m][t]["ppl_std"] for t in TARGETS]
        
        ax_a.plot(x_pts, y_pts, label=meta["label"], color=meta["color"],
                  marker=meta["marker"], linestyle=meta["linestyle"],
                  linewidth=meta["linewidth"], markersize=meta["markersize"],
                  zorder=meta["zorder"])
        
        if max(y_err) > 0.01:
            ax_a.fill_between(x_pts, np.array(y_pts) - np.array(y_err),
                             np.array(y_pts) + np.array(y_err),
                             color=meta["color"], alpha=0.18, zorder=meta["zorder"]-1)
            
    ax_a.invert_xaxis()
    ax_a.set_xticks(TARGETS)
    ax_a.set_xticklabels([f"{t} Experts\n(-{int((64-t)/64*100)}%)" for t in TARGETS], fontsize=10)
    ax_a.set_ylabel("WikiText-2 Test Perplexity ↓", fontsize=12, fontweight="bold")
    ax_a.set_title("A. Perplexity vs. Active Experts (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    ax_a.legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    
    # ---------------- Panel B: Causal Test KL vs Active Experts ----------------
    ax_b = fig.add_subplot(gs[0, 1])
    
    for m in plot_methods_order:
        meta = METHOD_META[m]
        x_pts = TARGETS
        y_pts = [stats[m][t]["kl_mean"] for t in TARGETS]
        y_err = [stats[m][t]["kl_std"] for t in TARGETS]
        
        ax_b.plot(x_pts, y_pts, label=meta["label"], color=meta["color"],
                  marker=meta["marker"], linestyle=meta["linestyle"],
                  linewidth=meta["linewidth"], markersize=meta["markersize"],
                  zorder=meta["zorder"])
        
        if max(y_err) > 0.005:
            ax_b.fill_between(x_pts, np.array(y_pts) - np.array(y_err),
                             np.array(y_pts) + np.array(y_err),
                             color=meta["color"], alpha=0.18, zorder=meta["zorder"]-1)
            
    ax_b.invert_xaxis()
    ax_b.set_xticks(TARGETS)
    ax_b.set_xticklabels([f"{t} Experts\n(-{int((64-t)/64*100)}%)" for t in TARGETS], fontsize=10)
    ax_b.set_ylabel("Causal Held-Out KL Divergence (nats) ↓", fontsize=12, fontweight="bold")
    ax_b.set_title("B. Functional Output Divergence vs. Experts (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax_b.grid(True, linestyle="--", alpha=0.6)
    ax_b.legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    
    # ---------------- Panel C: Deep Compression Breakdown (48E / -25%) ----------------
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Sort methods by 48E PPL ascending
    methods_sorted_48e = sorted(plot_methods_order, key=lambda m: stats[m][48]["ppl_mean"])
    y_pos = np.arange(len(methods_sorted_48e))
    ppl_48 = [stats[m][48]["ppl_mean"] for m in methods_sorted_48e]
    ppl_err_48 = [stats[m][48]["ppl_std"] for m in methods_sorted_48e]
    bar_colors = [METHOD_META[m]["color"] for m in methods_sorted_48e]
    labels_48 = [METHOD_META[m]["label"] for m in methods_sorted_48e]
    
    bars = ax_c.barh(y_pos, ppl_48, xerr=ppl_err_48, color=bar_colors, alpha=0.88,
                     edgecolor="#333333", linewidth=1.0, capsize=4, height=0.62)
    
    # Highlight CARE-Adaptive bar
    for idx, m in enumerate(methods_sorted_48e):
        if m == "care_adaptive":
            bars[idx].set_edgecolor("#0D47A1")
            bars[idx].set_linewidth(2.2)
            
    ax_c.axvline(BASE_PPL, color="#424242", linestyle=":", linewidth=1.5,
                 label=f"64E Base ({BASE_PPL:.2f})")
    
    # Annotate values
    for idx, (p, m) in enumerate(zip(ppl_48, methods_sorted_48e)):
        delta = p - BASE_PPL
        ann_text = f" {p:.2f} (+{delta:.1f})"
        ax_c.text(p + (ppl_err_48[idx] if ppl_err_48[idx] > 0 else 0) + 0.4, idx,
                  ann_text, va="center", fontsize=9.5, fontweight="bold" if m=="care_adaptive" else "normal")
        
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels_48, fontsize=10.5)
    ax_c.invert_yaxis()  # Best on top
    ax_c.set_xlabel("WikiText-2 Test Perplexity at 48 Experts (-25%) ↓", fontsize=11, fontweight="bold")
    ax_c.set_xlim(0, 42)
    ax_c.set_title("C. Deep Compression Ranking at 48 Experts (-25% Capacity)", fontsize=13, fontweight="bold", pad=10)
    ax_c.grid(True, axis="x", linestyle="--", alpha=0.6)
    
    # ---------------- Panel D: Pareto Trade-off: Time vs Quality at 48E ----------------
    ax_d = fig.add_subplot(gs[1, 1])
    
    label_positions = {
        "hc_smoe": (1.3, 31.0, 1.4, 31.3),
        "mc_smoe": (13.6, 26.74, 15.0, 26.74),
        "sub_moe": (45.2, 34.84, 50.0, 34.84),
        "reap_prune": (12.9, 13.01, 14.5, 13.01),
        "care_adaptive": (1276.5, 19.88, 1400.0, 19.6),
        "care_static": (960.8, 24.33, 260.0, 25.2),
        "random": (919.8, 22.33, 270.0, 21.6),
    }
    
    for m in plot_methods_order:
        meta = METHOD_META[m]
        x_time = stats[m][48]["time_mean"]
        y_ppl = stats[m][48]["ppl_mean"]
        
        # Plot bubble/scatter point
        size = 150 if m == "care_adaptive" else 100
        ax_d.scatter(x_time, y_ppl, color=meta["color"], marker=meta["marker"],
                     s=size, edgecolors="#111111", linewidths=1.2, zorder=meta["zorder"],
                     label=meta["label"])
        
        # Annotation text
        tx, ty = label_positions[m][2], label_positions[m][3]
        short_name = meta["label"].split()[0]
        ax_d.text(tx, ty, short_name, fontsize=10,
                  fontweight="bold" if m=="care_adaptive" else "normal",
                  color=meta["color"], va="center")
        
    ax_d.set_xscale("log")
    ax_d.set_xlabel("Compression Wall-Clock Time (seconds, log scale) →", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("WikiText-2 Test Perplexity at 48E ↓", fontsize=11, fontweight="bold")
    ax_d.set_title("D. Efficiency-Quality Pareto Trade-off at 48 Experts", fontsize=13, fontweight="bold", pad=10)
    ax_d.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_d.set_xlim(0.8, 4500)
    ax_d.set_ylim(8, 38)
    
    # Overall Title
    fig.suptitle("OLMoE-1B-7B Multi-Seed Benchmark Master Results (NVIDIA L40S, Seeds 42–44)\nCanonical WikiText-2 Test Split (65,536 tokens, 128 chunks × 512, Top-8 Routing)",
                 fontsize=15, fontweight="bold", y=0.98)
    
    save_path = os.path.join(PLOTS_DIR, "fig_olmoe_master_benchmark_4panel.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    # Copy to artifacts directory
    artifact_path = os.path.join(ARTIFACT_DIR, "fig_olmoe_master_benchmark_4panel.png")
    shutil.copy2(save_path, artifact_path)
    print(f"Copied to artifact: {artifact_path}")


# ---------------------------------------------------------
# 2. Standalone Figure 1: Test PPL vs Active Experts
# ---------------------------------------------------------
def create_standalone_ppl_figure():
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
    
    # Reference baseline line
    ax.axhline(BASE_PPL, color="#333333", linestyle=":", linewidth=1.8, zorder=2,
               label=f"64E Uncompressed Base ({BASE_PPL:.2f})")
    
    plot_methods_order = ["reap_prune", "care_adaptive", "mc_smoe", "care_static", "random", "hc_smoe", "sub_moe"]
    for m in plot_methods_order:
        meta = METHOD_META[m]
        x_pts = TARGETS
        y_pts = [stats[m][t]["ppl_mean"] for t in TARGETS]
        y_err = [stats[m][t]["ppl_std"] for t in TARGETS]
        
        ax.plot(x_pts, y_pts, label=meta["label"], color=meta["color"],
                marker=meta["marker"], linestyle=meta["linestyle"],
                linewidth=meta["linewidth"], markersize=meta["markersize"],
                zorder=meta["zorder"])
        
        if max(y_err) > 0.01:
            ax.fill_between(x_pts, np.array(y_pts) - np.array(y_err),
                            np.array(y_pts) + np.array(y_err),
                            color=meta["color"], alpha=0.18, zorder=meta["zorder"]-1)
            
    ax.invert_xaxis()
    ax.set_xticks(TARGETS)
    ax.set_xticklabels([f"{t} Experts\n(-{int((64-t)/64*100)}% capacity)" for t in TARGETS], fontsize=11)
    ax.set_ylabel("WikiText-2 Test Perplexity (Lower is Better) ↓", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Surviving Experts per Layer (Compression $\\rightarrow$)", fontsize=12, fontweight="bold")
    ax.set_title("OLMoE-1B-7B: Language-Model Quality Under Expert Consolidation\n(Mean $\\pm$ 1 SD across Seeds 42, 43, 44 on NVIDIA L40S)",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10.5)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "fig1_ppl_vs_experts.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    artifact_path = os.path.join(ARTIFACT_DIR, "fig1_ppl_vs_experts.png")
    shutil.copy2(save_path, artifact_path)
    print(f"Copied to artifact: {artifact_path}")


# ---------------------------------------------------------
# 3. Standalone Figure 2: Causal Held-Out KL Divergence
# ---------------------------------------------------------
def create_standalone_kl_figure():
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
    
    plot_methods_order = ["reap_prune", "care_adaptive", "mc_smoe", "care_static", "random", "hc_smoe", "sub_moe"]
    for m in plot_methods_order:
        meta = METHOD_META[m]
        x_pts = TARGETS
        y_pts = [stats[m][t]["kl_mean"] for t in TARGETS]
        y_err = [stats[m][t]["kl_std"] for t in TARGETS]
        
        ax.plot(x_pts, y_pts, label=meta["label"], color=meta["color"],
                marker=meta["marker"], linestyle=meta["linestyle"],
                linewidth=meta["linewidth"], markersize=meta["markersize"],
                zorder=meta["zorder"])
        
        if max(y_err) > 0.005:
            ax.fill_between(x_pts, np.array(y_pts) - np.array(y_err),
                            np.array(y_pts) + np.array(y_err),
                            color=meta["color"], alpha=0.18, zorder=meta["zorder"]-1)
            
    ax.invert_xaxis()
    ax.set_xticks(TARGETS)
    ax.set_xticklabels([f"{t} Experts\n(-{int((64-t)/64*100)}% capacity)" for t in TARGETS], fontsize=11)
    ax.set_ylabel("Causal Held-Out Test KL Divergence (nats) ↓", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Surviving Experts per Layer (Compression $\\rightarrow$)", fontsize=12, fontweight="bold")
    ax.set_title("OLMoE-1B-7B: Functional Output Distribution Shift During Consolidation\n(Causally Aligned Next-Token Logits on 8,192 Held-Out Tokens)",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10.5)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "fig2_kl_vs_experts.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    artifact_path = os.path.join(ARTIFACT_DIR, "fig2_kl_vs_experts.png")
    shutil.copy2(save_path, artifact_path)
    print(f"Copied to artifact: {artifact_path}")


# ---------------------------------------------------------
# 4. Standalone Figure 3: Deep Compression Breakdown (48 Experts)
# ---------------------------------------------------------
def create_standalone_48e_breakdown():
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=300)
    
    plot_methods_order = ["reap_prune", "care_adaptive", "random", "care_static", "mc_smoe", "hc_smoe", "sub_moe"]
    methods_sorted_48e = sorted(plot_methods_order, key=lambda m: stats[m][48]["ppl_mean"])
    y_pos = np.arange(len(methods_sorted_48e))
    ppl_48 = [stats[m][48]["ppl_mean"] for m in methods_sorted_48e]
    ppl_err_48 = [stats[m][48]["ppl_std"] for m in methods_sorted_48e]
    bar_colors = [METHOD_META[m]["color"] for m in methods_sorted_48e]
    labels_48 = [METHOD_META[m]["label"] for m in methods_sorted_48e]
    
    bars = ax.barh(y_pos, ppl_48, xerr=ppl_err_48, color=bar_colors, alpha=0.88,
                   edgecolor="#222222", linewidth=1.0, capsize=5, height=0.60)
    
    # Highlight CARE-Adaptive
    for idx, m in enumerate(methods_sorted_48e):
        if m == "care_adaptive":
            bars[idx].set_edgecolor("#0D47A1")
            bars[idx].set_linewidth(2.6)
            
    ax.axvline(BASE_PPL, color="#333333", linestyle=":", linewidth=1.8,
               label=f"64E Uncompressed Base ({BASE_PPL:.2f})")
    
    # Add annotations
    for idx, (p, m) in enumerate(zip(ppl_48, methods_sorted_48e)):
        delta = p - BASE_PPL
        err_val = ppl_err_48[idx]
        if err_val > 0.01:
            ann_text = f" {p:.2f} ± {err_val:.2f} (+{delta:.1f})"
        else:
            ann_text = f" {p:.2f} (+{delta:.1f})"
        ax.text(p + err_val + 0.4, idx, ann_text, va="center", fontsize=10.5,
                fontweight="bold" if m=="care_adaptive" else "normal")
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_48, fontsize=11, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("WikiText-2 Test Perplexity at 48 Experts (-25% Capacity) ↓", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 44)
    ax.set_title("Deep Compression Performance at 48 Experts (-25% Expert Capacity)\nStatistical Margin of CARE-Adaptive Over External Merging Baselines ($p < 0.01$)",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    
    # Custom legend for Paradigms in upper right (where x > 25 is totally clear)
    legend_elements = [
        Patch(facecolor="#1565C0", edgecolor="#0D47A1", label="CARE-Adaptive (Ours - Merging)", linewidth=1.5),
        Patch(facecolor="#42A5F5", label="CARE-Static (Merging)"),
        Patch(facecolor="#FF7043", label="MC-SMoE (Stages 1-4)"),
        Patch(facecolor="#8E24AA", label="HC-SMoE (Router-proxy)"),
        Patch(facecolor="#D32F2F", label="Sub-MoE (Joint SVD)"),
        Patch(facecolor="#78909C", label="Random-K5 (Control)"),
        Patch(facecolor="#2E7D32", label="REAP-Prune (Pruning)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95, fontsize=9.5, title="Method Paradigm")
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "fig3_deep_compression_48e_bars.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    artifact_path = os.path.join(ARTIFACT_DIR, "fig3_deep_compression_48e_bars.png")
    shutil.copy2(save_path, artifact_path)
    print(f"Copied to artifact: {artifact_path}")


# ---------------------------------------------------------
# 5. Standalone Figure 4: Pareto Time vs Quality Trade-off
# ---------------------------------------------------------
def create_standalone_pareto_figure():
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
    
    plot_methods_order = ["reap_prune", "care_adaptive", "mc_smoe", "care_static", "random", "hc_smoe", "sub_moe"]
    
    label_positions = {
        "hc_smoe": (1.3, 31.0, 1.45, 31.4),
        "mc_smoe": (13.6, 26.74, 15.2, 26.74),
        "sub_moe": (45.2, 34.84, 51.0, 34.84),
        "reap_prune": (12.9, 13.01, 14.5, 13.01),
        "care_adaptive": (1276.5, 19.88, 1420.0, 19.7),
        "care_static": (960.8, 24.33, 260.0, 25.2),
        "random": (919.8, 22.33, 270.0, 21.6),
    }
    
    for m in plot_methods_order:
        meta = METHOD_META[m]
        x_time = stats[m][48]["time_mean"]
        y_ppl = stats[m][48]["ppl_mean"]
        
        size = 170 if m == "care_adaptive" else 120
        ax.scatter(x_time, y_ppl, color=meta["color"], marker=meta["marker"],
                   s=size, edgecolors="#111111", linewidths=1.3, zorder=meta["zorder"],
                   label=meta["label"])
        
        tx, ty = label_positions[m][2], label_positions[m][3]
        short_name = meta["label"]
        ax.text(tx, ty, short_name, fontsize=10.5,
                fontweight="bold" if m=="care_adaptive" else "medium",
                color=meta["color"], va="center")
        
    ax.axhline(BASE_PPL, color="#333333", linestyle=":", linewidth=1.5,
               label=f"64E Base ({BASE_PPL:.2f})")
    
    ax.set_xscale("log")
    ax.set_xlabel("Compression Wall-Clock Time (seconds, log scale) →", fontsize=12, fontweight="bold")
    ax.set_ylabel("WikiText-2 Test Perplexity at 48 Experts (-25%) ↓", fontsize=12, fontweight="bold")
    ax.set_title("OLMoE-1B-7B: Efficiency vs. Language-Model Quality Trade-off (48 Experts)\nFast Heuristic Proxies Collapse Under Deep Compression",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_xlim(0.8, 5500)
    ax.set_ylim(8, 38)
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "fig4_pareto_time_vs_ppl.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    artifact_path = os.path.join(ARTIFACT_DIR, "fig4_pareto_time_vs_ppl.png")
    shutil.copy2(save_path, artifact_path)
    print(f"Copied to artifact: {artifact_path}")


# ---------------------------------------------------------
# Run all generators
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Generating flagship 4-panel master benchmark figure...")
    create_4panel_master_figure()
    
    print("Generating standalone PPL figure...")
    create_standalone_ppl_figure()
    
    print("Generating standalone KL figure...")
    create_standalone_kl_figure()
    
    print("Generating standalone 48E breakdown figure...")
    create_standalone_48e_breakdown()
    
    print("Generating standalone Pareto trade-off figure...")
    create_standalone_pareto_figure()
    
    print("\nAll plots successfully generated and mirrored to artifact directory!")
