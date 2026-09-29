import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def main():
    print("="*50)
    print("PHASE 9: PLOTTING ALL BENCHMARK CHARTS (WITH REAP)")
    print("="*50)
    
    result_path = "benchmark_results/JetMoE-8B/final_benchmark.json"
    with open(result_path) as f:
        results = json.load(f)

    methods   = ["Random", "Sub-MoE", "REAP", "CARE Adaptive"]
    colors    = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]  # Grey, Blue, Orange, Green
    markers   = ["o",       "s",       "D",      "^"]
    checkpoints = [8, 7, 6, 4]
    base_ppl  = results["Base"]["8"]

    # ------------------------------------------------------------------ #
    # PLOT 1 — Line chart (log scale)                                     #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color, marker in zip(methods, colors, markers):
        ppls = [base_ppl] + [results[method][str(c)] for c in [7, 6, 4]]
        ax.plot(checkpoints, ppls, label=method, color=color,
                marker=marker, linewidth=2.5, markersize=8)

    ax.set_yscale("log")
    ax.set_xlabel("MLP Experts per Layer", fontsize=13)
    ax.set_ylabel("Perplexity (log scale, lower is better)", fontsize=13)
    ax.set_title("JetMoE-8B MLP Expert Compression — PPL Curve", fontsize=14, pad=15)
    ax.set_xticks(checkpoints)
    ax.invert_xaxis()
    ax.axhline(base_ppl, color="red", linestyle=":", linewidth=1.5, label=f"Base PPL ({base_ppl:.2f})")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("benchmark_results/JetMoE-8B/compression_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: compression_plot.png")

    # ------------------------------------------------------------------ #
    # PLOT 2 — Grouped bar chart (log scale)                              #
    # ------------------------------------------------------------------ #
    checkpoints_str = ["7", "6", "4"]
    x = np.arange(len(checkpoints_str))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (method, color) in enumerate(zip(methods, colors)):
        ppls   = [results[method][c] for c in checkpoints_str]
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, ppls, width, label=method, color=color,
                        edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.1f}" if h < 100 else f"{int(h)}"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_yscale("log")
    ax.set_ylabel("Perplexity (log scale, lower is better)", fontsize=12)
    ax.set_xlabel("MLP Experts per Layer", fontsize=12)
    ax.set_title("JetMoE-8B Compression Benchmark: Method Comparison", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints_str, fontsize=11)
    ax.axhline(base_ppl, color="red", linestyle=":", linewidth=1.5, label=f"Base PPL ({base_ppl:.2f})")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("benchmark_results/JetMoE-8B/compression_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: compression_barplot.png")

    # ------------------------------------------------------------------ #
    # PLOT 3 — Performance Retention bar chart (higher is better)         #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (method, color) in enumerate(zip(methods, colors)):
        retentions = [(base_ppl / results[method][c]) * 100 for c in checkpoints_str]
        offset     = (i - 1.5) * width
        bars       = ax.bar(x + offset, retentions, width, label=method, color=color,
                            edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.1f}%" if h >= 0.1 else "<0.1%"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Performance Retention (%, higher is better)", fontsize=12)
    ax.set_xlabel("MLP Experts per Layer", fontsize=12)
    ax.set_title("JetMoE-8B Performance Retention (Higher is Better)", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints_str, fontsize=11)
    ax.axhline(100, color="red", linestyle=":", linewidth=1.5, label="100% (Base)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("benchmark_results/JetMoE-8B/retention_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: retention_barplot.png")

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
