"""
care_large_moe_systems/plot_systems_results.py
Generates publication-quality systems benchmark plots:
- Storage & VRAM scaling vs expert count
- TTFT and ITL latency scaling vs expert count
- Prefill and Decode throughput scaling
- Routing entropy and load imbalance
- Quality (Perplexity) vs efficiency trade-offs
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "systems_benchmark_summary.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

def set_style():
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

def plot_memory_and_storage(df: pd.DataFrame, model_name: str = "olmoe"):
    sub = df[df["model"] == model_name].sort_values("experts", ascending=False)
    if sub.empty:
        return
    
    unique_exp = sub.drop_duplicates(subset=["experts"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Plot 1: Total Parameters & BF16 Checkpoint Size
    color1 = "#1f77b4"
    color2 = "#ff7f0e"
    
    ax1.plot(unique_exp["experts"], unique_exp["total_params_b"], marker="o", linewidth=2.2, color=color1, label="Total Parameters (B)")
    ax1.plot(unique_exp["experts"], unique_exp["active_params_b"], marker="s", linewidth=2.0, linestyle="--", color="#2ca02c", label="Active Parameters / Token (B)")
    ax1.set_xlabel("Number of Available Experts (E)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Parameter Count (Billions)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{model_name.upper()}: Parameter Scaling vs. Expert Count", fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    
    # Plot 2: Checkpoint Footprint (GB) vs Peak VRAM
    ax2.plot(unique_exp["experts"], unique_exp["size_bf16_gb"], marker="^", linewidth=2.2, color=color2, label="Unquantized BF16 Footprint (GB)")
    if "peak_vram_mb" in unique_exp.columns:
        ax2.plot(unique_exp["experts"], unique_exp["peak_vram_mb"] / 1024.0, marker="d", linewidth=2.0, linestyle="-.", color="#d62728", label="Peak Inference VRAM (GB)")
    ax2.set_xlabel("Number of Available Experts (E)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Memory (GB)", fontsize=11, fontweight="bold")
    ax2.set_title(f"{model_name.upper()}: Memory & Storage Footprint", fontsize=12, fontweight="bold")
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{model_name}_memory_storage_scaling.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Saved] {out_path}")

def plot_latency_and_throughput(df: pd.DataFrame, model_name: str = "olmoe"):
    sub = df[(df["model"] == model_name) & (df["batch_size"] == 1)].sort_values("experts", ascending=False)
    if sub.empty:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Plot 1: Inter-Token Latency (ITL) across Workloads
    for w_id in sub["workload"].unique():
        w_sub = sub[sub["workload"] == w_id]
        ax1.plot(w_sub["experts"], w_sub["mean_itl_ms"], marker="o", linewidth=2.0, label=f"Workload {w_id}")
    ax1.set_xlabel("Number of Available Experts (E)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Inter-Token Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{model_name.upper()}: Generation Latency (Batch Size = 1)", fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    
    # Plot 2: Decode Throughput (tok/s)
    for w_id in sub["workload"].unique():
        w_sub = sub[sub["workload"] == w_id]
        ax2.plot(w_sub["experts"], w_sub["decode_throughput_tok_sec"], marker="s", linewidth=2.0, label=f"Workload {w_id}")
    ax2.set_xlabel("Number of Available Experts (E)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Decode Throughput (tokens/sec)", fontsize=11, fontweight="bold")
    ax2.set_title(f"{model_name.upper()}: Sustained Decode Throughput", fontsize=12, fontweight="bold")
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{model_name}_latency_throughput.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Saved] {out_path}")

def plot_routing_dynamics(df: pd.DataFrame, model_name: str = "olmoe"):
    sub = df[df["model"] == model_name].drop_duplicates(subset=["experts"]).sort_values("experts", ascending=False)
    if sub.empty or "routing_entropy" not in sub.columns:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Plot 1: Routing Entropy
    ax1.plot(sub["experts"], sub["routing_entropy"], marker="o", color="#9467bd", linewidth=2.2)
    ax1.set_xlabel("Number of Available Experts (E)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Normalized Entropy (0 to 1)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{model_name.upper()}: Routing Entropy vs Expert Count", fontsize=12, fontweight="bold")
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True)
    
    # Plot 2: Load Imbalance CV
    ax2.plot(sub["experts"], sub["routing_imbalance_cv"], marker="s", color="#8c564b", linewidth=2.2)
    ax2.set_xlabel("Number of Available Experts (E)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Expert Load Imbalance (CV = std/mean)", fontsize=11, fontweight="bold")
    ax2.set_title(f"{model_name.upper()}: Load Imbalance Coefficient of Variation", fontsize=12, fontweight="bold")
    ax2.grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{model_name}_routing_dynamics.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot Saved] {out_path}")

def generate_all_plots():
    set_style()
    if not os.path.exists(SUMMARY_CSV):
        print(f"Summary CSV not found at {SUMMARY_CSV}. Cannot generate plots yet.")
        return
        
    df = pd.read_csv(SUMMARY_CSV)
    for model in df["model"].unique():
        plot_memory_and_storage(df, model)
        plot_latency_and_throughput(df, model)
        plot_routing_dynamics(df, model)

if __name__ == "__main__":
    generate_all_plots()
