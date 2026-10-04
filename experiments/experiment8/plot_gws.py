import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    return np.percentile(corrs, [2.5, 97.5])

def partial_corr(x, y, cov):
    slope_x, intercept_x, _, _, _ = stats.linregress(cov, x)
    res_x = x - (slope_x * cov + intercept_x)
    
    slope_y, intercept_y, _, _, _ = stats.linregress(cov, y)
    res_y = y - (slope_y * cov + intercept_y)
    
    return stats.spearmanr(res_x, res_y)[0]

def out_of_sample_r2(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return r2_score(y, y_pred)

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
    # Since noise_floor was 0.0000000 in FP32, any strictly positive damage is above noise floor.
    # We will use 1e-7 as a practical FP32 machine epsilon bound for sum reduction.
    noise_floor = 1e-7 
    within_noise = np.sum(np.abs(damage) <= noise_floor)
    above_noise = np.sum(np.abs(damage) > noise_floor)
    
    print(f"Number of negative ΔCE pairs: {neg_ce}")
    print(f"Minimum ΔCE: {min_ce:.7f}")
    print(f"Number within noise floor (abs(ΔCE) <= {noise_floor}): {within_noise}")
    print(f"Number above noise floor: {above_noise}")
    print(f"Total pairs evaluated: {len(damage)}")

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
        
    print("\n--- Partial Correlations ---")
    print(f"GWS | Usage:   {partial_corr(gws, damage, usage):.3f}")
    print(f"GWS | RW-L2:   {partial_corr(gws, damage, rw_l2):.3f}")
    print(f"RW-L2 | Usage: {partial_corr(rw_l2, damage, usage):.3f}")
    print(f"Usage | RW-L2: {partial_corr(usage, damage, rw_l2):.3f}")

    print("\n--- Incremental Out-of-Sample Models (GroupKFold over j) ---")
    # Features need to be 2D arrays
    X_usage = usage.reshape(-1, 1)
    X_usage_rw = np.column_stack((usage, rw_l2))
    X_usage_rw_gws = np.column_stack((usage, rw_l2, gws_sq)) # Using gws_sq since it's stronger than signed
    
    oos_r2_1 = out_of_sample_r2(X_usage, damage, j_arr)
    oos_r2_2 = out_of_sample_r2(X_usage_rw, damage, j_arr)
    oos_r2_3 = out_of_sample_r2(X_usage_rw_gws, damage, j_arr)
    
    print(f"1. Usage                 (Out-of-sample R²): {oos_r2_1:.4f}")
    print(f"2. Usage + RW-L2         (Out-of-sample R²): {oos_r2_2:.4f}")
    print(f"3. Usage + RW-L2 + GWS²  (Out-of-sample R²): {oos_r2_3:.4f}")

if __name__ == "__main__":
    main()
