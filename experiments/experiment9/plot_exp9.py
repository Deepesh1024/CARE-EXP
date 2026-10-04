import json
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clustered_bootstrap_rho(y_pred, y_true, groups, n_boot=10000, seed=42):
    np.random.seed(seed)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    groups = np.array(groups)
    
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    boot_rhos = []
    
    for _ in range(n_boot):
        # Sample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=n_groups, replace=True)
        
        # Build bootstrap sample indices
        boot_idx = []
        for g in sampled_groups:
            # Find all observations belonging to group g
            g_idx = np.where(groups == g)[0]
            boot_idx.extend(g_idx)
            
        boot_idx = np.array(boot_idx)
        
        # Calculate metric
        rho, _ = stats.spearmanr(y_pred[boot_idx], y_true[boot_idx])
        if not np.isnan(rho):
            boot_rhos.append(rho)
            
    if not boot_rhos:
        return np.nan, np.nan, np.nan
        
    boot_rhos = np.array(boot_rhos)
    mean_rho = np.mean(boot_rhos)
    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)
    
    return mean_rho, ci_lower, ci_upper

def main():
    results_path = "experiments/experiment9/results/fold_results.json"
    with open(results_path, "r") as f:
        data = json.load(f)
        
    # We will pool the predictions across all folds in Partition 00 for the main clustered bootstrap
    # (Since Partition 00 folds cover exactly the 2016 pairs once)
    
    models = ["Model 0", "Model 1", "Model 2", "Model 3", "Model 4"]
    model_names = {
        "Model 0": "Local-11 (Exp 4 Baseline)",
        "Model 1": "Usage + RW-L2",
        "Model 2": "CARE Geometry Only",
        "Model 3": "CARE + Usage",
        "Model 4": "CARE + functional controls"
    }
    
    print(f"{'Model':<35} | {'Spearman rho':<20} | {'OOS R2':<20}")
    print("-" * 80)
    
    summary_data = []
    
    for m in models:
        # Collect all predictions for partition 00
        p00_preds = []
        p00_true = []
        p00_groups = []
        
        fold_r2s = []
        
        for fold_res in data[m]:
            if fold_res["partition"] == 0:
                p00_preds.extend(fold_res["pred"])
                p00_true.extend(fold_res["y_true"])
                p00_groups.extend(fold_res["source_expert_j"])
                fold_r2s.append(fold_res["r2"])
                
        # Calculate pooled spearman and clustered bootstrap CI
        true_rho, _ = stats.spearmanr(p00_preds, p00_true)
        boot_mean, ci_low, ci_high = clustered_bootstrap_rho(p00_preds, p00_true, p00_groups)
        
        # Calculate mean OOS R2 across folds
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        
        summary_data.append({
            "Model": model_names[m],
            "rho": true_rho,
            "rho_ci_low": ci_low,
            "rho_ci_high": ci_high,
            "r2_mean": mean_r2,
            "r2_std": std_r2
        })
        
        print(f"{model_names[m]:<35} | {true_rho:.3f} [{ci_low:.3f}, {ci_high:.3f}] | {mean_r2:.3f} ± {std_r2:.3f}")
        
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [d["Model"].replace(" (", "\n(") for d in summary_data]
    rhos = [d["rho"] for d in summary_data]
    errs = [[d["rho"] - d["rho_ci_low"] for d in summary_data], 
            [d["rho_ci_high"] - d["rho"] for d in summary_data]]
            
    ax1.bar(names, rhos, yerr=errs, capsize=5, color=sns.color_palette("viridis", 5))
    ax1.set_title("Spearman Rank Correlation (Clustered 95% CI)")
    ax1.set_ylabel("Spearman $\\rho$")
    ax1.tick_params(axis='x', rotation=45)
    
    r2s = [d["r2_mean"] for d in summary_data]
    r2_errs = [d["r2_std"] for d in summary_data]
    
    ax2.bar(names, r2s, yerr=r2_errs, capsize=5, color=sns.color_palette("mako", 5))
    ax2.set_title("Out-of-Sample $R^2$ (5-Fold Mean ± Std)")
    ax2.set_ylabel("$R^2$")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("experiments/experiment9/results/model_comparison.png", dpi=300)
    
    # Save markdown table
    with open("experiments/experiment9/results/table.md", "w") as f:
        f.write("| Predictor | Spearman $\\rho$ | Bootstrap 95% CI | OOS $R^2$ (Mean ± Std) |\n")
        f.write("|-----------|----------------|------------------|----------------------|\n")
        for d in summary_data:
            f.write(f"| {d['Model']} | {d['rho']:.3f} | [{d['rho_ci_low']:.3f}, {d['rho_ci_high']:.3f}] | {d['r2_mean']:.3f} ± {d['r2_std']:.3f} |\n")

if __name__ == "__main__":
    main()
