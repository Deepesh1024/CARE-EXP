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

# 1. Performance Retention Bar Chart (Higher is better)
fig, ax = plt.subplots(figsize=(10, 6))

ax.axhline(y=100.0, color='black', linestyle='--', linewidth=2, label='Baseline (16 Experts, 100%)')

for i, (method, config) in enumerate(methods_to_plot.items()):
    label, color = config
    y_vals = []
    for t in targets:
        if t in results[method]:
            mean_ppl = np.mean(results[method][t])
            # Performance Retention = (Baseline PPL / Compressed PPL) * 100
            retention = (baseline_ppl / mean_ppl) * 100
            y_vals.append(retention)
        else:
            y_vals.append(0)
    
    offset = (i - 1) * width
    ax.bar(x + offset, y_vals, width, label=label, color=color, edgecolor='black')

ax.set_title("Phi-3.5-MoE: Performance Retention", fontsize=16, fontweight="bold")
ax.set_xlabel("Number of Surviving Experts", fontsize=14)
ax.set_ylabel("Predictive Probability Retention (%)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.set_ylim(0, 110)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("experiments/experiment12/results/plots/retention_bar_chart.png", dpi=300)
plt.close()

# 2. PPL Degradation Bar Chart (Lower is better, but maybe they just wanted retention)
# I'll just make the retention one the primary one.
print("Plots generated!")
