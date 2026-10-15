"""
build_master_benchmark_v2.py
Compiles the master CSVs, JSONs, and generates high-resolution comparative figures
for Qwen3-30B-A3B on WikiText-2 across all merging-based compression methods (excluding REAP).
"""

import os
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/complete_comparative_benchmark_v2"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
ARTIFACTS_DIR = "/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/8d6d78d9-760d-4f2e-bbee-c8d7644d95c5"

EXPERT_COUNTS = [128, 112, 96, 80, 64, 60]

# Verified WikiText-2 PPL (Single Model: Qwen3-30B-A3B, Single Dataset: WikiText-2 test set, 32k tokens)
WIKITEXT_PPL = {
    "Baseline": {128: 9.5715},
    "CARE-Adaptive": {112: 11.1902, 96: 14.8101, 80: 19.9292, 64: 31.3531, 60: 33.5058},
    "HC-SMoE": {112: 14.4954, 96: 22.8044, 80: 38.3396, 64: 64.2068, 60: 74.0461},
    "Random": {112: 12.7422, 96: 26.3504, 80: 47.6214, 64: 189.8233, 60: 213.4241},
    "Sub-MoE": {112: 11.8771, 96: 14.3825, 80: 90.7231, 64: 176.9348, 60: 203.6507},
    "MC-SMoE": {112: 15.4906, 96: 82.1218, 80: 235.3179, 64: 490.4447, 60: 542.8728},
    "REAP": {112: 11.2011, 96: 13.1724, 80: 16.1392, 64: 22.0596, 60: 23.9455} # Kept in CSV/JSON docs, omitted from merging plots/report
}

# Top-k Sweeps on WikiText-2 at 80 Experts (and Baseline at 128 Experts)
TOPK_PPL = {
    "Baseline-128": {8: 9.5715, 6: 9.8657, 4: 11.4332, 2: 52.6092},
    "CARE-80": {8: 19.9292, 6: 20.3620, 4: 24.4178, 2: 114.2377},
    "HC-SMoE-80": {8: 38.3396, 6: 38.4146, 4: 46.1562, 2: 207.6673},
    "MC-SMoE-80": {8: 235.3179, 6: 218.4853, 4: 224.5419, 2: 304.5240}
}

# Systems Specs
STORAGE_GB = {128: 59.70, 112: 52.95, 96: 46.19, 80: 39.44, 64: 32.69, 60: 31.00}
PARAMS_B = {128: 32.05, 112: 28.43, 96: 24.80, 80: 21.18, 64: 17.55, 60: 16.64}
VRAM_ALLOC_GB = {128: 5.89, 112: 5.71, 96: 5.57, 80: 36.62, 64: 29.87, 60: 28.18}
PEAK_VRAM_GB = {128: 5.89, 112: 5.71, 96: 5.57, 80: 40.53, 64: 33.74, 60: 32.05}
PREFILL_TOK_S = {128: 851.4, 112: 824.9, 96: 1078.9, 80: 12623.6, 64: 20196.0, 60: 20425.1}
DECODE_TOK_S = {128: 0.68, 112: 0.76, 96: 0.85, 80: 19.29, 64: 19.62, 60: 19.86}

# Compression Costs (to 80 Experts) - Merging Methods Only
COMPRESSION_COSTS_MERGING = [
    {"method": "CARE-Adaptive", "wall_clock_sec": 21148.8, "wall_clock_formatted": "5h 52m", "gpu_vram_gb": 0.0, "host_ram_gb": 40.2, "calibration_passes": 240, "algorithm_family": "Geometric Subspace Dynamic Merging", "in_ram_merging": "Yes"},
    {"method": "Sub-MoE", "wall_clock_sec": 533.8, "wall_clock_formatted": "8m 54s", "gpu_vram_gb": 0.0, "host_ram_gb": 40.2, "calibration_passes": 0, "algorithm_family": "Saliency-Guided Centroid Clustering", "in_ram_merging": "Yes"},
    {"method": "Random Merging", "wall_clock_sec": 475.7, "wall_clock_formatted": "7m 56s", "gpu_vram_gb": 0.0, "host_ram_gb": 40.2, "calibration_passes": 0, "algorithm_family": "Stochastic Uniform Pairing", "in_ram_merging": "Yes"},
    {"method": "HC-SMoE", "wall_clock_sec": 126.8, "wall_clock_formatted": "2m 07s", "gpu_vram_gb": 0.0, "host_ram_gb": 40.2, "calibration_passes": 0, "algorithm_family": "Hierarchical Agglomerative Clustering (Lance-Williams)", "in_ram_merging": "Yes"},
    {"method": "MC-SMoE", "wall_clock_sec": 78.9, "wall_clock_formatted": "1m 19s", "gpu_vram_gb": 0.0, "host_ram_gb": 40.2, "calibration_passes": 0, "algorithm_family": "Router-Guided Multi-Centroid Merging", "in_ram_merging": "Yes"},
]

# Reasoning Results (GSM8K / MATH-500) - Merging Methods
REASONING_RESULTS_MERGING = [
    {"model": "Qwen3-30B-A3B", "config": "Baseline (128E, k=8)", "gsm8k_acc": 20.0, "math500_acc": 0.0, "status": "Audited Baseline", "profile": "Standard reasoning baseline (Correct CoT steps)"},
    {"model": "Qwen3-30B-A3B", "config": "CARE-80 (80E, k=8)", "gsm8k_acc": 22.0, "math500_acc": 2.0, "status": "Audited Exp14", "profile": "Preserves reasoning geometry; solves discrete algebra"},
    {"model": "Qwen3-30B-A3B", "config": "Random-80 (80E, k=8)", "gsm8k_acc": 28.0, "math500_acc": 4.0, "status": "Audited Exp14", "profile": "Stochastic variance preservation prevents circuit loss"},
    {"model": "Qwen3-30B-A3B", "config": "Sub-MoE-80 (80E, k=8)", "gsm8k_acc": 23.0, "math500_acc": 0.0, "status": "Audited Exp14", "profile": "Basic CoT logic intact despite language modeling degradation"},
]

# --- 2. Write master_results.csv (Comprehensive Documentation) ---
csv_path = os.path.join(OUTPUT_DIR, "master_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "dataset", "eval_token_budget", "method", "method_type", "expert_count", "inference_topk", 
        "regime", "wikitext_ppl", "delta_ppl_vs_base", "rel_ppl_pct", 
        "total_params_b", "active_params_b", "checkpoint_size_gb", 
        "prefill_tok_sec", "decode_tok_sec", "vram_allocated_gb", "peak_vram_gb"
    ])
    
    # Baseline 128
    for k in [8, 6, 4, 2]:
        ppl = TOPK_PPL["Baseline-128"][k]
        delta = ppl - 9.5715
        rel = f"+{(delta / 9.5715)*100:.2f}%"
        writer.writerow([
            "Qwen/Qwen3-30B-A3B", "WikiText-2", 32768, "Baseline", "Dense Uncompressed", 128, k,
            "Regime B (CPU-Offloaded)", f"{ppl:.4f}", f"{delta:.4f}", rel,
            32.05, 4.86, 59.70, 851.4, 0.68, 5.89, 5.89
        ])
        
    # Expert sweeps at k=8
    for method, scores in WIKITEXT_PPL.items():
        if method == "Baseline":
            continue
        m_type = "Parameter Pruning" if method == "REAP" else "Expert Merging"
        for e in [112, 96, 80, 64, 60]:
            if e in scores:
                ppl = scores[e]
                delta = ppl - 9.5715
                rel = f"+{(delta / 9.5715)*100:.2f}%"
                regime = "Regime B (CPU-Offloaded)" if e > 80 else "Regime A (Pure GPU-Resident)"
                pref = PREFILL_TOK_S[e]
                deco = DECODE_TOK_S[e]
                v_alloc = VRAM_ALLOC_GB[e]
                v_peak = PEAK_VRAM_GB[e]
                writer.writerow([
                    "Qwen/Qwen3-30B-A3B", "WikiText-2", 32768, method, m_type, e, 8,
                    regime, f"{ppl:.4f}", f"{delta:.4f}", rel,
                    PARAMS_B[e], 4.86, STORAGE_GB[e], pref, deco, v_alloc, v_peak
                ])
                
    # Top-k sweeps at 80E
    for m_label, k_dict in [("CARE-Adaptive", TOPK_PPL["CARE-80"]), ("HC-SMoE", TOPK_PPL["HC-SMoE-80"]), ("MC-SMoE", TOPK_PPL["MC-SMoE-80"])]:
        for k in [6, 4, 2]:
            ppl = k_dict[k]
            delta = ppl - 9.5715
            rel = f"+{(delta / 9.5715)*100:.2f}%"
            writer.writerow([
                "Qwen/Qwen3-30B-A3B", "WikiText-2", 32768, m_label, "Expert Merging", 80, k,
                "Regime A (Pure GPU-Resident)", f"{ppl:.4f}", f"{delta:.4f}", rel,
                21.18, 4.86, 39.44, PREFILL_TOK_S[80], DECODE_TOK_S[80], 36.62, 40.53
            ])

print(f"Generated {csv_path}")

# --- 3. Write compression_costs.csv ---
comp_csv_path = os.path.join(OUTPUT_DIR, "compression_costs.csv")
with open(comp_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "target_experts", "method", "wall_clock_sec", "wall_clock_formatted",
        "gpu_vram_gb", "host_ram_gb", "calibration_passes", "in_ram_merging", "algorithm_family"
    ])
    for item in COMPRESSION_COSTS_MERGING:
        writer.writerow([
            "Qwen/Qwen3-30B-A3B", 80, item["method"], item["wall_clock_sec"], item["wall_clock_formatted"],
            item["gpu_vram_gb"], item["host_ram_gb"], item["calibration_passes"], item["in_ram_merging"], item["algorithm_family"]
        ])
print(f"Generated {comp_csv_path}")

# --- 4. Write master_results.json ---
json_path = os.path.join(OUTPUT_DIR, "master_results.json")
master_dict = {
    "model": "Qwen/Qwen3-30B-A3B",
    "dataset": "WikiText-2 (32 chunks x 1024 tokens = 32,768 tokens, Salesforce/wikitext test split)",
    "single_dataset_single_model_guarantee": True,
    "wikitext_ppl_by_method": WIKITEXT_PPL,
    "topk_sweeps_ppl": TOPK_PPL,
    "hardware_regimes": {
        "Regime_A": {"threshold": "E <= 80", "vram_allocated_gb": "<= 36.62 GB", "decode_throughput_tok_sec": "~19.3 - 19.9 tok/s"},
        "Regime_B": {"threshold": "E > 80", "vram_allocated_gb": "5.57 - 5.89 GB (CPU streaming)", "decode_throughput_tok_sec": "~0.68 - 0.85 tok/s"}
    },
    "systems_profile": {
        "storage_gb": STORAGE_GB,
        "parameters_b": PARAMS_B,
        "vram_allocated_gb": VRAM_ALLOC_GB,
        "prefill_tok_sec": PREFILL_TOK_S,
        "decode_tok_sec": DECODE_TOK_S
    },
    "reasoning_evaluations_merging": REASONING_RESULTS_MERGING,
    "compression_costs_merging": COMPRESSION_COSTS_MERGING
}
with open(json_path, "w") as f:
    json.dump(master_dict, f, indent=2)
print(f"Generated {json_path}")

# --- 5. Generate Merging-Only Publication Plots (NO REAP) ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = {
    "Baseline": "#1f77b4",
    "CARE-Adaptive": "#2ca02c",
    "HC-SMoE": "#ff7f0e",
    "Sub-MoE": "#9467bd",
    "Random": "#8c564b",
    "MC-SMoE": "#e377c2"
}
markers = {
    "Baseline": "o",
    "CARE-Adaptive": "s",
    "HC-SMoE": "D",
    "Sub-MoE": "v",
    "Random": "x",
    "MC-SMoE": "p"
}

# Plot 1: WikiText-2 PPL vs Retained Experts (Merging Methods Only)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axhline(y=9.5715, color='#1f77b4', linestyle='--', linewidth=2, label='Baseline (128E, pristine) = 9.57')

merging_methods = ["CARE-Adaptive", "HC-SMoE", "Random", "Sub-MoE", "MC-SMoE"]
for m in merging_methods:
    x_vals = sorted(list(WIKITEXT_PPL[m].keys()), reverse=True)
    y_vals = [WIKITEXT_PPL[m][x] for x in x_vals]
    ax.plot(x_vals, y_vals, label=m, color=colors[m], marker=markers[m], linewidth=2.4, markersize=8)

ax.set_yscale('log')
ax.set_xlabel("Retained Expert Population ($E$)", fontsize=13, fontweight='bold')
ax.set_ylabel("WikiText-2 Perplexity (Log Scale, 32k Tokens)", fontsize=13, fontweight='bold')
ax.set_title("Language Modeling Perplexity vs. Retained Experts (Merging Methods)\nModel: Qwen3-30B-A3B | Dataset: WikiText-2 (32,768 Tokens)", fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=80, color='gray', linestyle=':', linewidth=1.5)
ax.text(80.5, 350, 'Regime A (Pure GPU-Resident)\nE <= 80 (<= 39.4 GB)', fontsize=10, color='green', fontweight='bold')
ax.text(82, 350, 'Regime B (CPU-Streaming)\nE > 80 (>= 46.2 GB)', fontsize=10, color='darkred', fontweight='bold', ha='right')
ax.set_xticks([128, 112, 96, 80, 64, 60])
ax.legend(frameon=True, fontsize=10.5, loc='upper right')
plt.tight_layout()
p1_path = os.path.join(PLOTS_DIR, "wikitext_ppl_vs_experts_k8.png")
fig.savefig(p1_path)
plt.close(fig)
print(f"Saved {p1_path}")

# Plot 2: WikiText-2 PPL vs Inference Top-k Sparsity
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
k_vals = [8, 6, 4, 2]
ax.plot(k_vals, [TOPK_PPL["Baseline-128"][k] for k in k_vals], label="Baseline-128 (Regime B)", color="#1f77b4", marker="o", linewidth=2.2, markersize=8)
ax.plot(k_vals, [TOPK_PPL["CARE-80"][k] for k in k_vals], label="CARE-80 (Regime A)", color="#2ca02c", marker="s", linewidth=2.2, markersize=8)
ax.plot(k_vals, [TOPK_PPL["HC-SMoE-80"][k] for k in k_vals], label="HC-SMoE-80 (Regime A)", color="#ff7f0e", marker="D", linewidth=2.2, markersize=8)
ax.plot(k_vals, [TOPK_PPL["MC-SMoE-80"][k] for k in k_vals], label="MC-SMoE-80 (Regime A)", color="#e377c2", marker="p", linewidth=2.2, markersize=8)

ax.set_yscale('log')
ax.set_xlabel("Inference Top-$k$ Sparsity", fontsize=13, fontweight='bold')
ax.set_ylabel("WikiText-2 Perplexity (Log Scale)", fontsize=13, fontweight='bold')
ax.set_title("Inference Top-$K$ Sparsity Sweep (Qwen3-30B on WikiText-2)", fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(k_vals)
ax.invert_xaxis()
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
p2_path = os.path.join(PLOTS_DIR, "wikitext_ppl_vs_topk.png")
fig.savefig(p2_path)
plt.close(fig)
print(f"Saved {p2_path}")

# Plot 3: Quality vs Memory Pareto Frontier (Merging Methods Only)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
for m in merging_methods:
    x_mem = [STORAGE_GB[e] for e in sorted(WIKITEXT_PPL[m].keys())]
    y_ppl = [WIKITEXT_PPL[m][e] for e in sorted(WIKITEXT_PPL[m].keys())]
    ax.plot(x_mem, y_ppl, label=m, color=colors[m], marker=markers[m], linewidth=2.2)

ax.scatter([STORAGE_GB[128]], [9.5715], color='#1f77b4', s=130, label='Baseline (128E)', zorder=5)
ax.set_yscale('log')
ax.set_xlabel("Checkpoint Storage Footprint (GB)", fontsize=13, fontweight='bold')
ax.set_ylabel("WikiText-2 Perplexity (Log Scale)", fontsize=13, fontweight='bold')
ax.set_title("Pareto Frontier: Checkpoint Storage vs. Language Modeling Quality", fontsize=13, fontweight='bold', pad=15)
ax.axvline(x=48.0, color='red', linestyle='--', linewidth=1.5, label='48 GB L40S VRAM Hardware Boundary')
ax.legend(frameon=True, fontsize=9.5, loc='upper left')
plt.tight_layout()
p3_path = os.path.join(PLOTS_DIR, "pareto_frontier_ppl_vram.png")
fig.savefig(p3_path)
plt.close(fig)
print(f"Saved {p3_path}")

# Plot 4: Hardware Regimes Prefill & Decode Throughput
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
e_list = [128, 112, 96, 80, 64, 60]
prefill_list = [PREFILL_TOK_S[e] for e in e_list]
decode_list = [DECODE_TOK_S[e] for e in e_list]
regime_colors = ['#d95f02' if e > 80 else '#1b9e77' for e in e_list]

ax1.bar([str(e) for e in e_list], prefill_list, color=regime_colors, width=0.55)
ax1.set_xlabel("Retained Expert Population ($E$)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Prefill Throughput (Tokens / Sec)", fontsize=12, fontweight='bold')
ax1.set_title("Prefill Throughput (Regime B vs. Regime A)", fontsize=12, fontweight='bold')
ax1.set_yscale('log')

ax2.bar([str(e) for e in e_list], decode_list, color=regime_colors, width=0.55)
ax2.set_xlabel("Retained Expert Population ($E$)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Autoregressive Decode Throughput (Tokens / Sec)", fontsize=12, fontweight='bold')
ax2.set_title("Decode Throughput (28.4x Step Discontinuity)", fontsize=12, fontweight='bold')

ax2.text(2.5, 10, 'Regime A: Pure GPU Resident\n~19.3 - 19.9 tok/s\n(Zero PCIe Layer Copies)', fontsize=9, color='#1b9e77', fontweight='bold')
ax2.text(0.5, 2.5, 'Regime B:\nCPU-Offloaded Streaming\n~0.68 - 0.85 tok/s\n(PCIe Transfer Bound)', fontsize=9, color='#d95f02', fontweight='bold')

plt.tight_layout()
p4_path = os.path.join(PLOTS_DIR, "regime_systems_throughput.png")
fig.savefig(p4_path)
plt.close(fig)
print(f"Saved {p4_path}")

# Copy to artifacts directory
import shutil
for p in [p1_path, p2_path, p3_path, p4_path]:
    shutil.copy(p, ARTIFACTS_DIR)

print("All merging-only plots compiled and copied to artifacts successfully!")
