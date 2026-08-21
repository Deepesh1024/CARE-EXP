import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp6b")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Aesthetic settings
sns.set_theme(style="darkgrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

def load_data():
    with open(os.path.join(RESULTS_DIR, "telemetry", "tau_database.json"), "r") as f:
        tau_db = json.load(f)
    with open(os.path.join(RESULTS_DIR, "metrics", "exposure_displacement_models.json"), "r") as f:
        models_db = json.load(f)
    return tau_db, models_db

def plot_tau_vs_magnitude(tau_db):
    """Plot Scatter of Tau vs Displacement Magnitude"""
    layers = ["first", "middle", "last"]
    checkpoints = ["checkpoint_10", "checkpoint_40", "checkpoint_70"]
    next_checkpoints = {"checkpoint_10": "checkpoint_40", "checkpoint_40": "checkpoint_70", "checkpoint_70": "checkpoint_100"}
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), sharex=True, sharey=True)
    fig.suptitle("Expert Displacement Magnitude vs Top-K Routing Frequency", fontsize=24, y=0.95)
    
    for i, layer in enumerate(layers):
        for j, ckpt in enumerate(checkpoints):
            next_ckpt = next_checkpoints[ckpt]
            ax = axes[i, j]
            
            try:
                tau = np.array(tau_db[layer][ckpt]["macro"]["tau_topk"])
                deltaC_path = os.path.join(RESULTS_DIR, f"embeddings/q4/deltaC_{layer}_{ckpt}_to_{next_ckpt}.npy")
                deltaC = np.load(deltaC_path)
                magnitudes = np.linalg.norm(deltaC, axis=1)
                
                # Scatter
                sns.regplot(x=tau, y=magnitudes, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
                
                # Correlation
                rho, _ = spearmanr(tau, magnitudes)
                ax.set_title(f"{layer.title()} Layer: {ckpt.split('_')[1]}k \u2192 {next_ckpt.split('_')[1]}k\nSpearman \u03c1 = {rho:.3f}", fontsize=14)
                
                if i == 2:
                    ax.set_xlabel("Routing Frequency (\u03c4_TopK)")
                if j == 0:
                    ax.set_ylabel("Displacement Magnitude ||\u0394C||")
            except Exception as e:
                ax.set_title(f"Data Missing ({e})")
                
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(PLOTS_DIR, "tau_vs_magnitude.png"))
    plt.close()

def plot_r2_scores(models_db):
    """Plot R2 scores of predictive models"""
    data = []
    for layer, transitions in models_db.items():
        for item in transitions:
            trans = item["transition"]
            for m_name, m_data in item.items():
                if m_name in ["transition", "M0_Zero"]:
                    continue
                data.append({
                    "Layer": layer.title(),
                    "Transition": trans,
                    "Model": m_name,
                    "R2": m_data["r2"]
                })
                
    import pandas as pd
    df = pd.DataFrame(data)
    
    if df.empty:
        return
        
    g = sns.catplot(
        data=df, kind="bar",
        x="Transition", y="R2", hue="Model", col="Layer",
        height=6, aspect=1.2, palette="muted"
    )
    g.fig.suptitle("Predictive Model R\u00b2 Scores (Predicting Vector \u0394C)", y=1.05, fontsize=20)
    g.set_axis_labels("Training Transition", "R\u00b2 Score (Negative = Worse than Mean)")
    g.set_xticklabels(rotation=15)
    
    # Add horizontal line at 0
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
        
    plt.savefig(os.path.join(PLOTS_DIR, "predictive_r2_scores.png"), bbox_inches='tight')
    plt.close()

def main():
    print("Loading data...")
    tau_db, models_db = load_data()
    
    print("Generating plot: Tau vs Magnitude...")
    plot_tau_vs_magnitude(tau_db)
    
    print("Generating plot: Predictive Model R2 Scores...")
    plot_r2_scores(models_db)
    
    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
