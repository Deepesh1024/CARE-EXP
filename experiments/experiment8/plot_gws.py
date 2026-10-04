import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def bootstrap_ci(x, y, group, n_boot=1000):
    np.random.seed(42)
    corrs = []
    unique_groups = np.unique(group)
    for _ in range(n_boot):
        # Sample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(group == g)[0] for g in sampled_groups])
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            corrs.append(r)
    return np.percentile(corrs, [2.5, 97.5])

def partial_corr(x, y, cov):
    # Regress x and y on cov, then correlate residuals
    slope_x, intercept_x, _, _, _ = stats.linregress(cov, x)
    res_x = x - (slope_x * cov + intercept_x)
    
    slope_y, intercept_y, _, _, _ = stats.linregress(cov, y)
    res_y = y - (slope_y * cov + intercept_y)
    
    return stats.spearmanr(res_x, res_y)[0]

def main():
    results_file = "experiments/experiment8/results.json"
    if not os.path.exists(results_file):
        results_file = "experiments/experiment8/results_partial.json"
        
    print(f"Loading {results_file}...")
    with open(results_file, "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    j_arr = df['j'].values
    gws = df['gws'].values
    gws_sq = df['gws_squared'].values
    rw_l2 = df['rw_l2'].values
    usage = df['usage'].values
    damage = df['actual_damage'].values
    
    print("\n--- Correlations vs Actual Damage ---")
    metrics = [
        ("GWS Signed", gws),
        ("Usage", usage),
        ("RW-L2", rw_l2),
        ("GWS Squared", gws_sq)
    ]
    
    for name, vals in metrics:
        s_r, s_p = stats.spearmanr(vals, damage)
        ci = bootstrap_ci(vals, damage, j_arr)
        print(f"{name}: Spearman r = {s_r:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]), p={s_p:.2e}")
        
    partial_r = partial_corr(gws, damage, usage)
    print(f"\nPartial Spearman r (GWS vs Damage controlling for Usage): {partial_r:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (name, vals) in enumerate(metrics):
        axes[i].scatter(vals, damage, alpha=0.3, s=15)
        axes[i].set_title(f"{name} vs ΔCE (ρ={stats.spearmanr(vals, damage)[0]:.3f})")
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Actual ΔCE")
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    os.makedirs("experiments/experiment8/plots", exist_ok=True)
    out_path = "experiments/experiment8/plots/full_correlations.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved plots to {out_path}")

if __name__ == "__main__":
    main()
