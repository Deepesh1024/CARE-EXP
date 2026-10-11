import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

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
    "care_adaptive": ("CARE-Adaptive (Ours)", "#1f77b4"),
    "random": ("Random Merge", "#7f7f7f"),
    "reap": ("REAP (Pruning)", "#d62728")
}

targets = [14, 12, 10, 8]
baseline_ppl = np.mean(results["uncompressed"][14])

x = np.arange(len(targets))
width = 0.25

# 1. PPL Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

ax.axhline(y=baseline_ppl, color='black', linestyle='--', linewidth=2, label='Baseline (16 Experts)')

for i, (method, config) in enumerate(methods_to_plot.items()):
    label, color = config
    y_vals = []
    for t in targets:
        if t in results[method]:
            y_vals.append(np.mean(results[method][t]))
        else:
            y_vals.append(0)
    
    offset = (i - 1) * width
    ax.bar(x + offset, y_vals, width, label=label, color=color, edgecolor='black')

ax.set_title("Phi-3.5-MoE: Perplexity Comparison", fontsize=16, fontweight="bold")
ax.set_xlabel("Number of Surviving Experts", fontsize=14)
ax.set_ylabel("Perplexity (Lower is better)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("experiments/experiment12/results/ppl_bar_chart.png", dpi=300)
plt.close()

# 2. Degradation Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

for i, (method, config) in enumerate(methods_to_plot.items()):
    label, color = config
    y_vals = []
    for t in targets:
        if t in results[method]:
            y_vals.append(np.mean(results[method][t]) - baseline_ppl)
        else:
            y_vals.append(0)
    
    offset = (i - 1) * width
    ax.bar(x + offset, y_vals, width, label=label, color=color, edgecolor='black')

ax.set_title("Phi-3.5-MoE: Perplexity Degradation", fontsize=16, fontweight="bold")
ax.set_xlabel("Number of Surviving Experts", fontsize=14)
ax.set_ylabel("Absolute PPL Increase (Lower is better)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("experiments/experiment12/results/deg_bar_chart.png", dpi=300)
plt.close()

print("Bar charts generated!")
