import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

os.makedirs("benchmark_results/JetMoE-8B/plots_noreap", exist_ok=True)

with open("benchmark_results/JetMoE-8B/final_benchmark.json") as f:
    results = json.load(f)

methods   = ["Random", "Sub-MoE", "CARE Adaptive"]
colors    = ["#95a5a6", "#3498db", "#2ecc71"]
markers   = ["o",       "s",       "^"]
checkpoints = [8, 7, 6, 4]
base_ppl  = results["Base"]["8"]

# PLOT 1: Line Chart
fig, ax = plt.subplots(figsize=(10, 6))
for method, color, marker in zip(methods, colors, markers):
    ppls = [base_ppl] + [results[method][str(c)] for c in [7, 6, 4]]
    ax.plot(checkpoints, ppls, label=method, color=color, marker=marker, linewidth=2.5, markersize=8)

ax.set_yscale("log")
ax.set_xlabel("MLP Experts per Layer", fontsize=13)
ax.set_ylabel("Perplexity (log scale, lower is better)", fontsize=13)
ax.set_title("JetMoE-8B Compression - PPL Curve", fontsize=14, pad=15)
ax.set_xticks(checkpoints)
ax.invert_xaxis()
ax.axhline(base_ppl, color="red", linestyle=":", linewidth=1.5, label=f"Base ({base_ppl:.2f})")
ax.legend(fontsize=11)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("benchmark_results/JetMoE-8B/plots/compression_plot.png", dpi=300)
plt.close()

# PLOT 2: Grouped Bar
checkpoints_str = ["7", "6", "4"]https://127.0.0.1:54374/static/artifacts/23683cb2-7c60-4fdf-b154-f50c39443184/.user_uploaded/media_1790794680163.png?csrf=3849518a-8731-463d-b9bd-d5c967c7de3c
x = np.arange(len(checkpoints_str))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, (method, color) in enumerate(zip(methods, colors)):
    ppls = [results[method][c] for c in checkpoints_str]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, ppls, width, label=method, color=color, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        label = f"{h:.1f}" if h < 100 else f"{int(h)}"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

ax.set_yscale("log")
ax.set_ylabel("Perplexity (log scale, lower is better)", fontsize=12)
ax.set_xlabel("MLP Experts per Layer", fontsize=12)
ax.set_title("JetMoE-8B Method Comparison", fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(checkpoints_str, fontsize=11)
ax.axhline(base_ppl, color="red", linestyle=":", linewidth=1.5, label=f"Base ({base_ppl:.2f})")
ax.legend(fontsize=10)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("benchmark_results/JetMoE-8B/plots/compression_barplot.png", dpi=300)
plt.close()

# PLOT 3: Performance Retention (Higher is Better)
fig, ax = plt.subplots(figsize=(10, 6))
for i, (method, color) in enumerate(zip(methods, colors)):
    retentions = [(base_ppl / results[method][c]) * 100 for c in checkpoints_str]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, retentions, width, label=method, color=color, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        label = f"{h:.1f}%" if h >= 0.1 else "<0.1%"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

ax.set_ylim(0, 105)
ax.set_ylabel("Performance Retention (%, higher is better)", fontsize=12)
ax.set_xlabel("MLP Experts per Layer", fontsize=12)
ax.set_title("JetMoE-8B Performance Retention", fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(checkpoints_str, fontsize=11)
ax.axhline(100, color="red", linestyle=":", linewidth=1.5, label="100% (Base)")
ax.legend(fontsize=10)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("benchmark_results/JetMoE-8B/plots/retention_barplot.png", dpi=300)
plt.close()

