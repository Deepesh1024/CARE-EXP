import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("="*50)
    print("PHASE 9: PLOTTING BENCHMARK BAR PLOT")
    print("="*50)
    
    result_path = "benchmark_results/JetMoE-8B/final_benchmark.json"
    if not os.path.exists(result_path):
        print(f"Error: {result_path} not found. Run phase 8 first.")
        return
        
    with open(result_path, "r") as f:
        results = json.load(f)
        
    methods = ["Random", "Sub-MoE", "CARE Adaptive"]
    colors = ["#95a5a6", "#3498db", "#2ecc71"]  # Grey, Blue, Green
    
    # We want to plot the checkpoints 7, 6, 4
    checkpoints = ["7", "6", "4"]
    x = np.arange(len(checkpoints))
    width = 0.25  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        # Extract the PPL for this method at each checkpoint
        ppls = [results[method][str(ckpt)] for ckpt in checkpoints]
        
        # Calculate the x position for this method's bars
        offset = (i - 1) * width
        bars = ax.bar(x + offset, ppls, width, label=method, color=color, edgecolor='black', linewidth=1)
        
        # Add labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    # Styling the plot
    ax.set_ylabel('Perplexity (Log Scale, Lower is better)', fontsize=12)
    ax.set_xlabel('Number of Experts Remaining (out of 8)', fontsize=12)
    ax.set_title('JetMoE-8B Compression Benchmark: Method Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, fontsize=11)
    ax.legend(fontsize=11)
    
    # Use logarithmic scale because Random at 4 experts is ~5000 while CARE is ~87
    ax.set_yscale('log')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add base PPL as a reference line
    base_ppl = results["Base"]["8"]
    ax.axhline(y=base_ppl, color='red', linestyle=':', linewidth=2, label=f'Base PPL ({base_ppl:.2f})')
    ax.legend()

    plt.tight_layout()
    
    out_file = "benchmark_results/JetMoE-8B/compression_barplot.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Bar plot saved successfully to {out_file}!")
    
if __name__ == "__main__":
    main()
