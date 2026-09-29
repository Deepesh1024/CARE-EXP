import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("="*50)
    print("PHASE 9: PLOTTING RETENTION BAR PLOT (HIGHER IS BETTER)")
    print("="*50)
    
    result_path = "benchmark_results/JetMoE-8B/final_benchmark.json"
    if not os.path.exists(result_path):
        print(f"Error: {result_path} not found.")
        return
        
    with open(result_path, "r") as f:
        results = json.load(f)
        
    methods = ["Random", "Sub-MoE", "CARE Adaptive"]
    colors = ["#95a5a6", "#3498db", "#2ecc71"]  # Grey, Blue, Green
    
    checkpoints = ["7", "6", "4"]
    x = np.arange(len(checkpoints))
    width = 0.25  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    base_ppl = results["Base"]["8"]
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        # Calculate Performance Retention Ratio = (Base PPL / Method PPL) * 100
        # Higher is better. 100% means no degradation.
        retentions = [(base_ppl / results[method][str(ckpt)]) * 100 for ckpt in checkpoints]
        
        # Calculate the x position for this method's bars
        offset = (i - 1) * width
        bars = ax.bar(x + offset, retentions, width, label=method, color=color, edgecolor='black', linewidth=1)
        
        # Add labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            # If height is very small (< 1%), show 1 decimal point, otherwise just integer
            label_text = f'{height:.1f}%' if height < 10 else f'{int(height)}%'
            ax.annotate(label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Styling the plot
    ax.set_ylabel('Performance Retention Ratio (%)\n(Higher is better, 100% = No Degradation)', fontsize=12)
    ax.set_xlabel('Number of Experts Remaining (out of 8)', fontsize=12)
    ax.set_title('JetMoE-8B Performance Retention (Higher is Better)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, fontsize=11)
    
    # We can use a linear scale here because it's bounded 0-100%, 
    # but a log scale might help see the tiny bars at 4 experts. 
    # Let's use a symmetric log scale or just linear with a high limit.
    # We will use linear for intuitive "%" reading, but add horizontal grid lines.
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a 100% reference line
    ax.axhline(y=100, color='red', linestyle=':', linewidth=2, label='100% (Base PPL)')
    
    # Position legend out of the way
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    
    out_file = "benchmark_results/JetMoE-8B/retention_barplot.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Retention bar plot saved successfully to {out_file}!")
    
if __name__ == "__main__":
    main()
