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
    
    sns.set_theme(style="whitegrid")
    
    # Figure 1: PPL vs number of experts
    plt.figure(figsize=(10, 6))
    # We use errorbar=None to remove fake uncertainty shading for deterministic methods
    sns.lineplot(data=df, x='experts', y='ppl', hue='method', marker='o', errorbar=None)
    plt.gca().invert_xaxis()
    plt.title("Language-model quality under expert consolidation", fontsize=14)
    plt.xlabel("Number of Active Experts", fontsize=12)
    plt.ylabel("WikiText-2 Perplexity ↓", fontsize=12)
    plt.legend(title="Method")
    plt.savefig(os.path.join(plots_dir, "fig1_ppl_vs_experts.png"), dpi=300)
    plt.close()
    
    # Dataframe filtered for methods that actually measured KL
    df_kl = df.dropna(subset=['cumulative_kl'])
    
    # Figure 2: Cumulative Functional Divergence During Compression
    plt.figure(figsize=(10, 6))
    if not df_kl.empty:
        sns.lineplot(data=df_kl, x='experts', y='cumulative_kl', hue='method', marker='s', errorbar=None)
    plt.gca().invert_xaxis()
    plt.title("Cumulative Functional Divergence During Compression", fontsize=14)
    plt.xlabel("Number of Active Experts", fontsize=12)
    plt.ylabel("Cumulative KL Divergence ↓", fontsize=12)
    plt.legend(title="Method")
    plt.savefig(os.path.join(plots_dir, "fig2_cum_kl_vs_experts.png"), dpi=300)
    plt.close()
    
    # Figure 3: Marginal Functional Damage per Compression Step
    df_kl = df_kl.copy()
    df_kl['step'] = 64 - df_kl['experts']
    plt.figure(figsize=(10, 6))
    
    df_sorted = df_kl.sort_values(['method', 'seed', 'step']).copy()
    # No fillna(0) here!
    df_sorted['marginal_kl'] = df_sorted.groupby(['method', 'seed'])['cumulative_kl'].diff()
    
    if not df_sorted[df_sorted['step'] > 0].empty:
        sns.lineplot(data=df_sorted[df_sorted['step'] > 0], x='step', y='marginal_kl', hue='method', marker='^', errorbar=None)
    plt.title("Marginal Functional Damage per Compression Step", fontsize=14)
    plt.xlabel("Compression Step", fontsize=12)
    plt.ylabel("Marginal KL Divergence ↓", fontsize=12)
    plt.legend(title="Method")
    plt.savefig(os.path.join(plots_dir, "fig3_marginal_kl.png"), dpi=300)
    plt.close()
    
    # Figure 4: Compression time vs number of experts removed
    df['step'] = 64 - df['experts']
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='compression_time_sec', hue='method', marker='x', errorbar=None)
    plt.title("Compression Time vs Experts Removed\n(Note: Wall-clock time is implementation-dependent)", fontsize=14)
    plt.xlabel("Experts Removed", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.legend(title="Method")
    plt.savefig(os.path.join(plots_dir, "fig4_time_vs_removed.png"), dpi=300)
    plt.close()
    
    # Figure 5: PPL degradation vs parameter reduction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='param_reduction_pct', y='delta_ppl', hue='method', marker='d', errorbar=None)
    plt.title("PPL Degradation vs Parameter Reduction", fontsize=14)
    plt.xlabel("Parameter Reduction (%)", fontsize=12)
    plt.ylabel("Delta PPL (from 64 experts) ↓", fontsize=12)
    plt.legend(title="Method")
    plt.savefig(os.path.join(plots_dir, "fig5_ppl_deg_vs_param_red.png"), dpi=300)
    plt.close()
    
    # Figure 6: Functional damage vs parameter reduction
    plt.figure(figsize=(10, 6))
    if not df_kl.empty:
        sns.lineplot(data=df_kl, x='param_reduction_pct', y='cumulative_kl', hue='method', marker='*', errorbar=None)
    plt.title("Functional Damage vs Parameter Reduction", fontsize=14)
    plt.xlabel("Parameter Reduction (%)", fontsize=12)
    plt.ylabel("Cumulative KL Divergence ↓", fontsize=12)
    plt.legend(title="Method")
    plt.savefig(os.path.join(plots_dir, "fig6_func_damage_vs_param_red.png"), dpi=300)
    plt.close()
    
    # Figure 7: CARE candidate capability distance vs actual marginal KL
    selection_csv = os.path.join(summary_dir, "selection_analysis.csv")
    if os.path.exists(selection_csv):
        df_sel = pd.read_csv(selection_csv)
        if not df_sel.empty:
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_sel, x='capability_distance', y='actual_marginal_kl', hue='step', palette='viridis', s=100)
            plt.title("Observed Capability Distance and Intervention Damage", fontsize=14)
            plt.xlabel("Capability Distance (Geometric)", fontsize=12)
            plt.ylabel("Actual Marginal KL (Functional) ↓", fontsize=12)
            # Add caption text below plot
            plt.figtext(0.5, 0.01, "Descriptive visualization of observed selected interventions.", ha="center", fontsize=10, style='italic')
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(os.path.join(plots_dir, "fig7_cap_dist_vs_kl.png"), dpi=300)
            plt.close()
            
    print(f"Successfully generated all plots in {plots_dir}")

if __name__ == "__main__":
    benchmark_dir = os.path.join(os.path.dirname(__file__), "..", "benchmark_results")
    summary_dir = os.path.join(benchmark_dir, "summary")
    generate_plots(summary_dir)
