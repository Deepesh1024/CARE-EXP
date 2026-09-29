import json
import matplotlib.pyplot as plt
import os

def main():
    print("="*50)
    print("PHASE 9: PLOTTING BENCHMARK RESULTS")
    print("="*50)
    
    result_path = "benchmark_results/JetMoE-8B/final_benchmark.json"
    if not os.path.exists(result_path):
        print(f"Error: {result_path} not found. Run phase 8 first.")
        return
        
    with open(result_path, "r") as f:
        results = json.load(f)
        
    plt.figure(figsize=(10, 6))
    
    # We will plot the PPL degradation from Base PPL.
    base_ppl = results["Base"]["8"]
    
    methods = ["Random", "Sub-MoE", "CARE Adaptive"]
    colors = ["grey", "blue", "green"]
    markers = ["o", "s", "^"]
    
    for method, color, marker in zip(methods, colors, markers):
        data = results[method]
        # X is number of experts: 8 (base), 7, 6, 4
        # We need to sort by x
        x = [8]
        y = [base_ppl]
        
        for k, v in data.items():
            x.append(int(k))
            y.append(v)
            
        # Sort by x ascending
        xy = sorted(zip(x, y))
        x_sorted = [val[0] for val in xy]
        y_sorted = [val[1] for val in xy]
        
        plt.plot(x_sorted, y_sorted, marker=marker, color=color, linewidth=2, label=method)
        
    plt.gca().invert_xaxis() # 8 -> 4
    plt.title("JetMoE-8B Compression Benchmark (WikiText PPL)")
    plt.xlabel("Number of MLP Experts per Layer")
    plt.ylabel("Perplexity (Lower is better)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    out_file = "benchmark_results/JetMoE-8B/compression_plot.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to {out_file}!")
    
if __name__ == "__main__":
    main()
