"""
care_large_moe_systems/master_benchmark/plot_master_benchmark.py
Generates the complete set of master plots for the Qwen3-30B-A3B systems benchmark:
1. qwen3_ppl_vs_experts_k8.png: Perplexity vs. Expert Count (Methods at matched k=8)
2. qwen3_ppl_vs_topk.png: Perplexity vs. Top-K Routing Sparsity (Full vs. Pilot)
3. qwen3_pareto_frontiers.png: Pareto Frontiers (Quality vs Storage & Quality vs Throughput)
4. qwen3_topk_throughput.png: Prefill Throughput vs. Top-K Sparsity across Regimes
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "master_benchmark_results.csv")
os.makedirs(PLOTS_DIR, exist_ok=True)

def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 1.0

METHOD_STYLE = {
    "Baseline": {"color": "#111111", "marker": "o", "label": "Dense Baseline (128E)"},
    "CARE-Adaptive": {"color": "#1f77b4", "marker": "s", "label": "CARE-Adaptive (Merge)"},
    "REAP": {"color": "#2ca02c", "marker": "^", "label": "REAP (Prune)"},
    "Sub-MoE": {"color": "#ff7f0e", "marker": "D", "label": "Sub-MoE (Cluster)"},
    "Random": {"color": "#d62728", "marker": "v", "label": "Random Merging"}
}

def plot_ppl_vs_experts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    sub = df[(df["inference_topk"] == 8) & (df["eval_type"].str.startswith("Full")) & (df["validation_status"].str.startswith("Verified"))].copy()
    
    for method, style in METHOD_STYLE.items():
        m_df = sub[sub["method"] == method].sort_values("expert_count", ascending=False)
        if not m_df.empty:
            ax.plot(
                m_df["expert_count"],
                m_df["wikitext_ppl"],
                marker=style["marker"],
                color=style["color"],
                linewidth=2.2,
                markersize=7,
                label=style["label"]
            )
            
    ax.set_xlabel("Available Experts per Layer (E)")
    ax.set_ylabel("WikiText-2 Perplexity (Lower is Better)")
    ax.set_title("Perplexity vs. Expert Population Count (Qwen3-30B, Full 32k Tokens, k=8)")
    ax.set_ylim([8, 250])
    ax.set_yscale("log")
    ax.axvline(x=80, color="#888888", linestyle=":", linewidth=1.5, label="Hardware VRAM Boundary (48 GB L40S)")
    ax.annotate("Regime A: GPU-Resident (E ≤ 80)", xy=(72, 12), xytext=(62, 10), fontsize=9, color="#1b5e20", weight="bold")
    ax.annotate("Regime B: Offloaded (E > 80)", xy=(90, 12), xytext=(88, 10), fontsize=9, color="#b71c1c", weight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
    
    out_path = os.path.join(PLOTS_DIR, "qwen3_ppl_vs_experts_k8.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Generated] {out_path}")

def plot_ppl_vs_topk(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    # 1. Full 32k tokens: Baseline 128E & CARE 80E
    base_full = df[(df["method"] == "Baseline") & (df["eval_type"].str.startswith("Full"))].sort_values("inference_topk", ascending=False)
    care_full = df[(df["method"] == "CARE-Adaptive") & (df["expert_count"] == 80) & (df["eval_type"].str.startswith("Full"))].sort_values("inference_topk", ascending=False)
    
    # 2. Pilot 2k tokens
    base_pilot = df[(df["method"] == "Baseline-Pilot")].sort_values("inference_topk", ascending=False)
    care_pilot = df[(df["method"] == "CARE-Pilot")].sort_values("inference_topk", ascending=False)
    
    ax.plot(base_full["inference_topk"], base_full["wikitext_ppl"], marker="o", color="#111111", linewidth=2.4, label="Baseline 128E (Full WikiText-2, 32k tok)")
    ax.plot(care_full["inference_topk"], care_full["wikitext_ppl"], marker="s", color="#1f77b4", linewidth=2.4, label="CARE 80E (Full WikiText-2, 32k tok)")
    
    ax.plot(base_pilot["inference_topk"], base_pilot["wikitext_ppl"], marker="o", color="#777777", linestyle="--", linewidth=1.5, label="Baseline 128E (Pilot, 2k tok)")
    ax.plot(care_pilot["inference_topk"], care_pilot["wikitext_ppl"], marker="s", color="#6baed6", linestyle="--", linewidth=1.5, label="CARE 80E (Pilot, 2k tok)")

    ax.set_xlabel("Inference Active Experts per Token (Top-K Sparsity)")
    ax.set_ylabel("WikiText-2 Perplexity (Log Scale)")
    ax.set_yscale("log")
    ax.set_ylim([8, 200])
    ax.set_title("Perplexity vs. Inference Top-K Routing Sparsity (Control Experiment)")
    ax.set_xticks([8, 6, 4, 2])
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
    
    out_path = os.path.join(PLOTS_DIR, "qwen3_ppl_vs_topk.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Generated] {out_path}")

def plot_pareto_quality_efficiency(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    sub = df[(df["inference_topk"] == 8) & (df["eval_type"].str.startswith("Full")) & (df["validation_status"].str.startswith("Verified"))].copy()
    
    # Panel A: PPL vs Storage Footprint
    for method, style in METHOD_STYLE.items():
        m_df = sub[sub["method"] == method].sort_values("checkpoint_size_gb")
        if not m_df.empty:
            ax1.scatter(
                m_df["checkpoint_size_gb"],
                m_df["wikitext_ppl"],
                color=style["color"],
                marker=style["marker"],
                s=75,
                label=style["label"]
            )
            ax1.plot(m_df["checkpoint_size_gb"], m_df["wikitext_ppl"], color=style["color"], linestyle="--", alpha=0.6)
            for _, r in m_df.iterrows():
                ax1.annotate(f"{int(r['expert_count'])}E", (r["checkpoint_size_gb"] + 0.4, r["wikitext_ppl"]), fontsize=8)

    ax1.set_xlabel("Checkpoint Footprint (GB in BF16)")
    ax1.set_ylabel("WikiText-2 Perplexity (Log Scale)")
    ax1.set_yscale("log")
    ax1.set_ylim([8, 250])
    ax1.set_title("Pareto Frontier: Quality vs. Storage Footprint")
    ax1.axvline(x=45, color="#d62728", linestyle=":", label="48 GB L40S VRAM Threshold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    # Panel B: PPL vs Decode Throughput (Highlighting Regime Boundary)
    for method, style in METHOD_STYLE.items():
        m_df = sub[sub["method"] == method].sort_values("decode_tok_sec")
        if not m_df.empty:
            ax2.scatter(
                m_df["decode_tok_sec"],
                m_df["wikitext_ppl"],
                color=style["color"],
                marker=style["marker"],
                s=75,
                label=style["label"]
            )
            ax2.plot(m_df["decode_tok_sec"], m_df["wikitext_ppl"], color=style["color"], linestyle="--", alpha=0.6)

    ax2.set_xlabel("Sustained Autoregressive Decode Throughput (tok / sec)")
    ax2.set_ylabel("WikiText-2 Perplexity (Log Scale)")
    ax2.set_yscale("log")
    ax2.set_ylim([8, 250])
    ax2.set_title("Pareto Frontier: Quality vs. Inference Throughput")
    ax2.axvspan(0, 1.5, color="#ffdddd", alpha=0.35, label="Regime B (CPU-Offloaded Streaming: ~0.7 tok/s)")
    ax2.axvspan(18, 22, color="#ddffdd", alpha=0.35, label="Regime A (Pure GPU-Resident: ~20 tok/s)")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "qwen3_pareto_frontiers.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Generated] {out_path}")

def plot_topk_throughput(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    base_full = df[(df["method"] == "Baseline") & (df["eval_type"].str.startswith("Full"))].sort_values("inference_topk", ascending=False)
    care_full = df[(df["method"] == "CARE-Adaptive") & (df["expert_count"] == 80) & (df["eval_type"].str.startswith("Full"))].sort_values("inference_topk", ascending=False)
    
    ax.plot(care_full["inference_topk"], care_full["prefill_tok_sec"], marker="s", color="#1f77b4", linewidth=2.4, label="CARE-80 (Regime A: GPU-Resident Prefill tok/s)")
    ax.plot(base_full["inference_topk"], base_full["prefill_tok_sec"], marker="o", color="#111111", linewidth=2.4, label="Baseline-128 (Regime B: CPU-Offloaded Prefill tok/s)")
    
    ax.set_xlabel("Inference Active Experts per Token (Top-K)")
    ax.set_ylabel("Prefill Throughput (Tokens / Second)")
    ax.set_title("Prefill Throughput vs. Top-K Sparsity across Systems Regimes")
    ax.set_xticks([8, 6, 4, 2])
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="center left")
    
    out_path = os.path.join(PLOTS_DIR, "qwen3_topk_throughput.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Generated] {out_path}")

def generate_all_master_plots():
    set_plot_style()
    if not os.path.exists(RESULTS_CSV):
        print(f"Dataset not found at {RESULTS_CSV}")
        return
    df = pd.read_csv(RESULTS_CSV)
    plot_ppl_vs_experts(df)
    plot_ppl_vs_topk(df)
    plot_pareto_quality_efficiency(df)
    plot_topk_throughput(df)

if __name__ == "__main__":
    generate_all_master_plots()
