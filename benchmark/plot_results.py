import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(summary_dir):
    csv_path = os.path.join(summary_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run aggregate_results.py first.")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: results.csv is empty.")
        return
        
    plots_dir = os.path.join(summary_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # We will average the Random baseline across seeds
    df_agg = df.groupby(['method', 'experts']).mean(numeric_only=True).reset_index()
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Figure 1: PPL vs number of experts
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='experts', y='ppl', hue='method', marker='o')
    plt.gca().invert_xaxis()
    plt.title("Figure 1: Perplexity vs Number of Experts")
    plt.xlabel("Number of Experts")
    plt.ylabel("WikiText-2 Perplexity")
    plt.savefig(os.path.join(plots_dir, "fig1_ppl_vs_experts.png"), dpi=300)
    plt.close()
    
    # Figure 2: Cumulative KL vs number of experts
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='experts', y='cumulative_kl', hue='method', marker='s')
    plt.gca().invert_xaxis()
    plt.title("Figure 2: Cumulative KL vs Number of Experts")
    plt.xlabel("Number of Experts")
    plt.ylabel("Cumulative KL Divergence")
    plt.savefig(os.path.join(plots_dir, "fig2_cum_kl_vs_experts.png"), dpi=300)
    plt.close()
    
    # Figure 3: Mean marginal KL vs compression step
    # We need to compute step from experts (64 - experts)
    df['step'] = 64 - df['experts']
    plt.figure(figsize=(10, 6))
    # We don't have step-level marginal KL in the summary CSV directly, but we can compute diff of cumulative
    df_sorted = df.sort_values(['method', 'seed', 'step'])
    df_sorted['marginal_kl'] = df_sorted.groupby(['method', 'seed'])['cumulative_kl'].diff().fillna(0)
    
    sns.lineplot(data=df_sorted[df_sorted['step'] > 0], x='step', y='marginal_kl', hue='method', marker='^')
    plt.title("Figure 3: Mean Marginal KL vs Compression Step")
    plt.xlabel("Compression Step")
    plt.ylabel("Marginal KL Divergence")
    plt.savefig(os.path.join(plots_dir, "fig3_marginal_kl.png"), dpi=300)
    plt.close()
    
    # Figure 4: Compression time vs number of experts removed
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='compression_time_sec', hue='method', marker='x')
    plt.title("Figure 4: Compression Time vs Experts Removed")
    plt.xlabel("Experts Removed")
    plt.ylabel("Time (seconds)")
    plt.savefig(os.path.join(plots_dir, "fig4_time_vs_removed.png"), dpi=300)
    plt.close()
    
    # Figure 5: PPL degradation vs parameter reduction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='param_reduction_pct', y='delta_ppl', hue='method', marker='d')
    plt.title("Figure 5: PPL Degradation vs Parameter Reduction")
    plt.xlabel("Parameter Reduction (%)")
    plt.ylabel("Delta PPL (from 64 experts)")
    plt.savefig(os.path.join(plots_dir, "fig5_ppl_deg_vs_param_red.png"), dpi=300)
    plt.close()
    
    # Figure 6: Functional damage vs parameter reduction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='param_reduction_pct', y='cumulative_kl', hue='method', marker='*')
    plt.title("Figure 6: Functional Damage vs Parameter Reduction")
    plt.xlabel("Parameter Reduction (%)")
    plt.ylabel("Cumulative KL Divergence")
    plt.savefig(os.path.join(plots_dir, "fig6_func_damage_vs_param_red.png"), dpi=300)
    plt.close()
    
    # Figure 7: CARE candidate capability distance vs actual marginal KL
    selection_csv = os.path.join(summary_dir, "selection_analysis.csv")
    if os.path.exists(selection_csv):
        df_sel = pd.read_csv(selection_csv)
        if not df_sel.empty:
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_sel, x='capability_distance', y='actual_marginal_kl', hue='step', palette='viridis', s=100)
            plt.title("Figure 7: Capability Distance vs Actual Marginal KL")
            plt.xlabel("Capability Distance (Geometric)")
            plt.ylabel("Actual Marginal KL (Functional)")
            plt.savefig(os.path.join(plots_dir, "fig7_cap_dist_vs_kl.png"), dpi=300)
            plt.close()
            
    print(f"Successfully generated all plots in {plots_dir}")

if __name__ == "__main__":
    benchmark_dir = os.path.join(os.path.dirname(__file__), "..", "benchmark_results")
    summary_dir = os.path.join(benchmark_dir, "summary")
    generate_plots(summary_dir)
