import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "care_com_v22", "trajectories")
    
    with open(os.path.join(base_dir, "static_v1.json"), "r") as f:
        static = json.load(f)
        
    with open(os.path.join(base_dir, "adaptive_v21.json"), "r") as f:
        adaptive = json.load(f)
        
    random_traces = []
    for seed in [42, 1024, 2027]:
        with open(os.path.join(base_dir, f"random_seed_{seed}.json"), "r") as f:
            random_traces.append(json.load(f))
            
    return static, adaptive, random_traces

def plot_perplexity(static, adaptive, random_traces, save_dir):
    plt.figure(figsize=(8, 6))
    
    checkpoints = [64, 60, 56]
    
    # Static
    static_ppl = [static["ppl"][str(c)] for c in checkpoints]
    plt.plot(checkpoints, static_ppl, marker='o', linestyle='-', label="Static v1", color="orange", linewidth=2, markersize=8)
    
    # Adaptive
    adaptive_ppl = [adaptive["ppl"][str(c)] for c in checkpoints]
    plt.plot(checkpoints, adaptive_ppl, marker='s', linestyle='-', label="Adaptive v2.1", color="green", linewidth=2, markersize=8)
    
    # Random
    random_ppls = []
    for trace in random_traces:
        random_ppls.append([trace["ppl"][str(c)] for c in checkpoints])
    random_mean = np.mean(random_ppls, axis=0)
    random_std = np.std(random_ppls, axis=0)
    
    plt.errorbar(checkpoints, random_mean, yerr=random_std, marker='^', linestyle='--', label="Random (Mean \u00b1 std)", color="gray", linewidth=2, markersize=8, capsize=5)
    
    plt.gca().invert_xaxis()
    plt.title("Perplexity over Compression Steps", fontsize=14)
    plt.xlabel("Number of Experts", fontsize=12)
    plt.ylabel("WikiText Perplexity (Lower is Better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "perplexity_plot.png"), dpi=300)
    plt.close()

def plot_marginal_kl(static, adaptive, random_traces, save_dir):
    plt.figure(figsize=(10, 5))
    
    steps = [t["step"] for t in static["trace"]]
    
    static_kl = [t["functional_damage"] for t in static["trace"]]
    adaptive_kl = [t["functional_damage"] for t in adaptive["trace"]]
    
    plt.plot(steps, static_kl, marker='o', label="Static v1", color="orange", linewidth=2)
    plt.plot(steps, adaptive_kl, marker='s', label="Adaptive v2.1", color="green", linewidth=2)
    
    random_kls = []
    for trace in random_traces:
        random_kls.append([t["functional_damage"] for t in trace["trace"]])
    random_mean = np.mean(random_kls, axis=0)
    plt.plot(steps, random_mean, marker='^', linestyle='--', label="Random (Mean)", color="gray", linewidth=2)
    
    plt.title("Marginal KL Damage Per Step", fontsize=14)
    plt.xlabel("Compression Step (1 to 8)", fontsize=12)
    plt.ylabel("KL Divergence", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "marginal_kl_plot.png"), dpi=300)
    plt.close()

def plot_cumulative_kl(static, adaptive, random_traces, save_dir):
    plt.figure(figsize=(10, 5))
    
    steps = [t["step"] for t in static["trace"]]
    # Add step 0 for origin
    steps = [0] + steps
    
    static_kl = [0] + list(np.cumsum([t["functional_damage"] for t in static["trace"]]))
    adaptive_kl = [0] + list(np.cumsum([t["functional_damage"] for t in adaptive["trace"]]))
    
    plt.plot(steps, static_kl, marker='o', label="Static v1", color="orange", linewidth=2)
    plt.plot(steps, adaptive_kl, marker='s', label="Adaptive v2.1", color="green", linewidth=2)
    
    random_kls = []
    for trace in random_traces:
        random_kls.append([0] + list(np.cumsum([t["functional_damage"] for t in trace["trace"]])))
    random_mean = np.mean(random_kls, axis=0)
    plt.plot(steps, random_mean, marker='^', linestyle='--', label="Random (Mean)", color="gray", linewidth=2)
    
    plt.title("Cumulative KL Damage", fontsize=14)
    plt.xlabel("Compression Step (0 to 8)", fontsize=12)
    plt.ylabel("Cumulative KL Divergence", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_kl_plot.png"), dpi=300)
    plt.close()

def main():
    static, adaptive, random_traces = load_data()
    
    save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "care_com_v22", "figures")
    os.makedirs(save_dir, exist_ok=True)
    
    plot_perplexity(static, adaptive, random_traces, save_dir)
    plot_marginal_kl(static, adaptive, random_traces, save_dir)
    plot_cumulative_kl(static, adaptive, random_traces, save_dir)
    
    print(f"Figures saved to {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
