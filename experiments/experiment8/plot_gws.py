import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

def bootstrap_ci(x, y, group, n_boot=1000):
    np.random.seed(42)
    corrs = []
    unique_groups = np.unique(group)
    for _ in range(n_boot):
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(group == g)[0] for g in sampled_groups])
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            corrs.append(r)
    if not corrs: return [np.nan, np.nan]
    return np.percentile(corrs, [2.5, 97.5])

def partial_corr(x, y, cov):
    slope_x, intercept_x, _, _, _ = stats.linregress(cov, x)
    res_x = x - (slope_x * cov + intercept_x)
    slope_y, intercept_y, _, _, _ = stats.linregress(cov, y)
    res_y = y - (slope_y * cov + intercept_y)
    return stats.spearmanr(res_x, res_y)[0]

def bootstrap_partial_ci(x, y, cov, group, n_boot=1000):
    np.random.seed(42)
    corrs = []
    unique_groups = np.unique(group)
    for _ in range(n_boot):
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(group == g)[0] for g in sampled_groups])
        r = partial_corr(x[idx], y[idx], cov[idx])
        if not np.isnan(r):
            corrs.append(r)
    if not corrs: return [np.nan, np.nan]
    return np.percentile(corrs, [2.5, 97.5])

def eval_oos_model(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    r2_scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        r2_scores.append(r2_score(y[test_idx], y_pred))
    return np.array(r2_scores)

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
    
    print("\n--- Diagnostic Stats (Low-Damage Regime) ---")
    neg_ce = np.sum(damage < 0)
    min_ce = np.min(damage)
    noise_floor = 1e-7 
    within_noise = np.sum(np.abs(damage) <= noise_floor)
    above_noise = np.sum(np.abs(damage) > noise_floor)
    
    total_evaluated = len(damage)
    print(f"Number of evaluated substitutions: {total_evaluated}")
    print(f"Number of negative ΔCE pairs: {neg_ce} ({(neg_ce/total_evaluated)*100:.1f}%)")
    print(f"Minimum ΔCE: {min_ce:.7f}")
    print(f"Number within noise floor (abs(ΔCE) <= {noise_floor}): {within_noise}")
    print(f"Number above noise floor: {above_noise}")

    print("\n--- Primary Correlations Table ---")
    print(f"{'Predictor':<25} {'Spearman ρ':<12} {'Bootstrap 95% CI':<20}")
    print("-" * 57)
    
    metrics = [
        ("GWS signed", gws),
        ("Usage", usage),
        ("RW-L2", rw_l2),
        ("GWS squared", gws_sq)
    ]
    
    for name, vals in metrics:
        s_r, _ = stats.spearmanr(vals, damage)
        ci = bootstrap_ci(vals, damage, j_arr)
        print(f"{name:<25} {s_r:<12.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        
    print("\n--- Partial Correlations (with Clustered Bootstrap CI) ---")
    partials = [
        ("GWS | Usage", gws, usage),
        ("GWS | RW-L2", gws, rw_l2),
        ("RW-L2 | Usage", rw_l2, usage),
        ("Usage | RW-L2", usage, rw_l2)
    ]
    for name, x, cov in partials:
        r = partial_corr(x, damage, cov)
        ci = bootstrap_partial_ci(x, damage, cov, j_arr)
        print(f"{name:<25} {r:<12.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\n--- Incremental Out-of-Sample Models (GroupKFold over j) ---")
    X_usage = usage.reshape(-1, 1)
    X_usage_rw = np.column_stack((usage, rw_l2))
    X_usage_rw_gws = np.column_stack((usage, rw_l2, gws)) 
    X_usage_rw_gwssq = np.column_stack((usage, rw_l2, gws_sq)) 
    
    r2_A = eval_oos_model(X_usage, damage, j_arr)
    r2_B = eval_oos_model(X_usage_rw, damage, j_arr)
    r2_C = eval_oos_model(X_usage_rw_gws, damage, j_arr)
    r2_D = eval_oos_model(X_usage_rw_gwssq, damage, j_arr)
    
    print(f"Model A: Usage")
    print(f"  Folds: {np.round(r2_A, 4)}")
    print(f"  Mean ± Std: {r2_A.mean():.4f} ± {r2_A.std():.4f}\n")
    
    print(f"Model B: Usage + RW-L2")
    print(f"  Folds: {np.round(r2_B, 4)}")
    print(f"  Mean ± Std: {r2_B.mean():.4f} ± {r2_B.std():.4f}\n")
    
    print(f"Model C: Usage + RW-L2 + signed GWS")
    print(f"  Folds: {np.round(r2_C, 4)}")
    print(f"  Mean ± Std: {r2_C.mean():.4f} ± {r2_C.std():.4f}\n")
    
    print(f"Model D: Usage + RW-L2 + GWS²")
    print(f"  Folds: {np.round(r2_D, 4)}")
    print(f"  Mean ± Std: {r2_D.mean():.4f} ± {r2_D.std():.4f}\n")

if __name__ == "__main__":
    main()
