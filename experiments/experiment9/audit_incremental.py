import json
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import r2_score

# Load Exp 9 results
with open("experiments/experiment9/results/fold_results.json", "r") as f:
    results = json.load(f)

# M0 = Model 1 (Usage + RW-L2)
# M1 = Model 4 (CARE + Usage + RW-L2)
m0_folds = results["Model 1"]
m1_folds = results["Model 4"]

def compute_metrics(y_true, y_pred):
    rho, _ = stats.spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rho, r2

# 1. Fold-wise comparisons
print("==================================================")
print("Exp9 Incremental-Baseline Audit: M0 vs M1")
print("M0: Usage + RW-L2")
print("M1: Usage + RW-L2 + CARE")
print("==================================================\n")

delta_rhos = []
delta_r2s = []
m0_rhos = []
m1_rhos = []
m0_r2s = []
m1_r2s = []

print(f"{'Fold':<10} | {'M0 ρ':<8} | {'M1 ρ':<8} | {'Δρ':<8} | {'M0 R²':<8} | {'M1 R²':<8} | {'ΔR²':<8}")
print("-" * 75)

all_m0_preds = []
all_m1_preds = []
all_y_trues = []
all_groups = [] # using source_expert_j for clustering

for i in range(len(m0_folds)):
    f0 = m0_folds[i]
    f1 = m1_folds[i]
    
    y_true = np.array(f0["y_true"])
    p0 = np.array(f0["pred"])
    p1 = np.array(f1["pred"])
    
    rho0, r2_0 = compute_metrics(y_true, p0)
    rho1, r2_1 = compute_metrics(y_true, p1)
    
    d_rho = rho1 - rho0
    d_r2 = r2_1 - r2_0
    
    m0_rhos.append(rho0)
    m1_rhos.append(rho1)
    m0_r2s.append(r2_0)
    m1_r2s.append(r2_1)
    delta_rhos.append(d_rho)
    delta_r2s.append(d_r2)
    
    # Store for bootstrap
    all_m0_preds.extend(p0)
    all_m1_preds.extend(p1)
    all_y_trues.extend(y_true)
    
    # Generate unique group IDs per fold to prevent cross-fold mixing during bootstrap
    groups = [f"{f0['partition']}_{f0['fold']}_{j}" for j in f0["source_expert_j"]]
    all_groups.extend(groups)

    print(f"P{f0['partition']}F{f0['fold']:<6} | {rho0:8.3f} | {rho1:8.3f} | {d_rho:8.3f} | {r2_0:8.3f} | {r2_1:8.3f} | {d_r2:8.3f}")

print("-" * 75)
print(f"{'MEAN':<10} | {np.mean(m0_rhos):8.3f} | {np.mean(m1_rhos):8.3f} | {np.mean(delta_rhos):8.3f} | {np.mean(m0_r2s):8.3f} | {np.mean(m1_r2s):8.3f} | {np.mean(delta_r2s):8.3f}\n")

# 2. Clustered Bootstrap CIs for Deltas
def clustered_bootstrap(y_true, p0, p1, groups, n_boot=2000):
    unique_groups = np.unique(groups)
    boot_delta_rhos = []
    boot_delta_r2s = []
    
    y_true = np.array(y_true)
    p0 = np.array(p0)
    p1 = np.array(p1)
    groups = np.array(groups)
    
    for _ in range(n_boot):
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
        
        yb = y_true[idx]
        p0b = p0[idx]
        p1b = p1[idx]
        
        rho0b, r20b = compute_metrics(yb, p0b)
        rho1b, r21b = compute_metrics(yb, p1b)
        
        boot_delta_rhos.append(rho1b - rho0b)
        boot_delta_r2s.append(r21b - r20b)
        
    ci_rho = np.percentile(boot_delta_rhos, [2.5, 97.5])
    ci_r2 = np.percentile(boot_delta_r2s, [2.5, 97.5])
    return ci_rho, ci_r2

ci_rho, ci_r2 = clustered_bootstrap(all_y_trues, all_m0_preds, all_m1_preds, all_groups)

# 3. Paired Permutation Test
def paired_permutation(y_true, p0, p1, n_perm=5000):
    y_true = np.array(y_true)
    p0 = np.array(p0)
    p1 = np.array(p1)
    
    rho0_base, r20_base = compute_metrics(y_true, p0)
    rho1_base, r21_base = compute_metrics(y_true, p1)
    
    d_rho_obs = rho1_base - rho0_base
    d_r2_obs = r21_base - r20_base
    
    count_rho = 0
    count_r2 = 0
    
    for _ in range(n_perm):
        # Swap predictions with 50% probability per instance
        swap = np.random.binomial(1, 0.5, size=len(y_true))
        p0_perm = np.where(swap, p1, p0)
        p1_perm = np.where(swap, p0, p1)
        
        rho0p, r20p = compute_metrics(y_true, p0_perm)
        rho1p, r21p = compute_metrics(y_true, p1_perm)
        
        d_rho_perm = rho1p - rho0p
        d_r2_perm = r21p - r20p
        
        if d_rho_perm >= d_rho_obs:
            count_rho += 1
        if d_r2_perm >= d_r2_obs:
            count_r2 += 1
            
    p_rho = (count_rho + 1) / (n_perm + 1)
    p_r2 = (count_r2 + 1) / (n_perm + 1)
    return p_rho, p_r2

p_rho, p_r2 = paired_permutation(all_y_trues, all_m0_preds, all_m1_preds)

print("--- Clustered Bootstrap CIs (ΔM1 - M0) ---")
print(f"Δρ 95% CI: [{ci_rho[0]:.3f}, {ci_rho[1]:.3f}]")
print(f"ΔR² 95% CI: [{ci_r2[0]:.3f}, {ci_r2[1]:.3f}]\n")

print("--- Paired Permutation P-Values ---")
print(f"Δρ p-value: {p_rho:.5f}")
print(f"ΔR² p-value: {p_r2:.5f}")
