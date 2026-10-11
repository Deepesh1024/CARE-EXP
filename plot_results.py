import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

with open("experiments/experiment12/results/all_results.json", "r") as f:
    data = json.load(f)

# Aggregate results
results = defaultdict(lambda: defaultdict(list))
for run in data:
    method = run["method"]
    target = run["target"]
    ppl = run["ppl"]
    results[method][target].append(ppl)

methods_to_plot = {
    "care_adaptive": ("CARE-Adaptive (Ours)", "blue", "o", "-"),
    "reap": ("REAP (Pruning)", "red", "s", "-"),
    "random": ("Random Merge", "gray", "^", "-"),
    "uncompressed": ("Baseline 16-Expert", "black", "x", "--")
}

targets = [16, 14, 12, 10, 8]
baseline_ppl = np.mean(results["uncompressed"][14]) # baseline is constant

# Calculate averages
plot_data = {}
for method, config in methods_to_plot.items():
    label, color, marker, ls = config
    y_ppl = []
    y_deg = []
    for t in targets:
        if t == 16:
            val = baseline_ppl
        else:
            if t in results[method]:
                val = np.mean(results[method][t])
            else:
                val = np.nan
        y_ppl.append(val)
        y_deg.append(val - baseline_ppl)
    plot_data[method] = {"ppl": y_ppl, "deg": y_deg, "cfg": config}

# 1. PPL Plot
plt.figure(figsize=(8, 6))
for method, data_dict in plot_data.items():
    label, color, marker, ls = data_dict["cfg"]
    plt.plot(targets, data_dict["ppl"], label=label, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=8)

plt.gca().invert_xaxis()
plt.title("Phi-3.5-MoE: Perplexity vs Expert Capacity", fontsize=14, fontweight="bold")
plt.xlabel("Number of Surviving Experts (Lower is more compressed)", fontsize=12)
plt.ylabel("Perplexity (Lower is better)", fontsize=12)
plt.xticks(targets)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/23683cb2-7c60-4fdf-b154-f50c39443184/care_ppl_plot.png", dpi=300)
plt.close()

# 2. Degradation Plot
plt.figure(figsize=(8, 6))
for method, data_dict in plot_data.items():
    if method == "uncompressed": continue
    label, color, marker, ls = data_dict["cfg"]
    plt.plot(targets, data_dict["deg"], label=label, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=8)

plt.gca().invert_xaxis()
plt.title("Phi-3.5-MoE: Perplexity Degradation", fontsize=14, fontweight="bold")
plt.xlabel("Number of Surviving Experts", fontsize=12)
plt.ylabel("Absolute PPL Increase (Lower is better)", fontsize=12)
plt.xticks(targets)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/23683cb2-7c60-4fdf-b154-f50c39443184/care_degradation_plot.png", dpi=300)
plt.close()

print("Plots generated!")
