import json
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score

def compute_metrics(y_true, y_pred):
    rho, _ = stats.spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rho, r2

with open("experiments/experiment9/results/fold_results.json", "r") as f:
    results = json.load(f)

m0_folds = results["Model 1"] # Usage + RW-L2
m1_folds = results["Model 4"] # CARE + Usage + RW-L2

# 1. Fold-wise comparisons
print("==================================================")
print("Exp9 Incremental-Baseline Audit: M0 vs M1 (Corrected)")
print("==================================================\n")

delta_rhos = []
delta_r2s = []
m0_rhos = []
m1_rhos = []
m0_r2s = []
m1_r2s = []

all_m0_preds = []
all_m1_preds = []
all_y_trues = []
all_groups = [] 

print("--- Fold-Wise Metrics ---")
print(f"{'Fold':<10} | {'M0 ρ':<8} | {'M1 ρ':<8} | {'Δρ':<8} | {'M0 R²':<8} | {'M1 R²':<8} | {'ΔR²':<8}")
print("-" * 75)

for i in range(len(m0_folds)):
    f0 = m0_folds[i]
    f1 = m1_folds[i]
    
    y_true = np.array(f0["y_true"])
    p0 = np.array(f0["pred"])
    p1 = np.array(f1["pred"])
    
    rho0, r2_0 = compute_metrics(y_true, p0)
    rho1, r2_1 = compute_metrics(y_true, p1)
    
    m0_rhos.append(rho0)
    m1_rhos.append(rho1)
    m0_r2s.append(r2_0)
    m1_r2s.append(r2_1)
    delta_rhos.append(rho1 - rho0)
    delta_r2s.append(r2_1 - r2_0)
    
    all_m0_preds.extend(p0)
    all_m1_preds.extend(p1)
    all_y_trues.extend(y_true)
    
    # Correct group construction: use the raw expert ID
    # This forms 64 unique source-expert clusters globally
    groups = f0["source_expert_j"]
    all_groups.extend(groups)

    print(f"P{f0['partition']}F{f0['fold']:<6} | {rho0:8.3f} | {rho1:8.3f} | {rho1-rho0:8.3f} | {r2_0:8.3f} | {r2_1:8.3f} | {r2_1-r2_0:8.3f}")

print("-" * 75)
print(f"{'MEAN':<10} | {np.mean(m0_rhos):8.3f} | {np.mean(m1_rhos):8.3f} | {np.mean(delta_rhos):8.3f} | {np.mean(m0_r2s):8.3f} | {np.mean(m1_r2s):8.3f} | {np.mean(delta_r2s):8.3f}")
print(f"{'MEDIAN':<10} | {np.median(m0_rhos):8.3f} | {np.median(m1_rhos):8.3f} | {np.median(delta_rhos):8.3f} | {np.median(m0_r2s):8.3f} | {np.median(m1_r2s):8.3f} | {np.median(delta_r2s):8.3f}\n")

all_m0_preds = np.array(all_m0_preds)
all_m1_preds = np.array(all_m1_preds)
all_y_trues = np.array(all_y_trues)
all_groups = np.array(all_groups)

unique_clusters = np.unique(all_groups)
obs_per_cluster = [np.sum(all_groups == g) for g in unique_clusters]

print(f"--- Dependency Structure ---")
print(f"Total predictions pooled: {len(all_y_trues)}")
print(f"Unique Source-Expert Clusters: {len(unique_clusters)}")
print(f"Observations per cluster: Min {np.min(obs_per_cluster)}, Max {np.max(obs_per_cluster)}, Mean {np.mean(obs_per_cluster):.1f}\n")

# 3. Pooled OOS Metrics
pooled_rho0, pooled_r20 = compute_metrics(all_y_trues, all_m0_preds)
pooled_rho1, pooled_r21 = compute_metrics(all_y_trues, all_m1_preds)
pooled_d_rho = pooled_rho1 - pooled_rho0
pooled_d_r2 = pooled_r21 - pooled_r20

print("--- Pooled OOS Metrics (Primary Estimands) ---")
print(f"Pooled M0 ρ:  {pooled_rho0:.3f}")
print(f"Pooled M1 ρ:  {pooled_rho1:.3f}")
print(f"Pooled Δρ:    {pooled_d_rho:.3f}")
print(f"Pooled M0 R²: {pooled_r20:.3f}")
print(f"Pooled M1 R²: {pooled_r21:.3f}")
print(f"Pooled ΔR²:   {pooled_d_r2:.3f}\n")

# 4. Genuinely Source-Expert-Clustered Bootstrap
def clustered_bootstrap(y_true, p0, p1, groups, n_boot=2000):
    unique_groups = np.unique(groups)
    boot_delta_rhos = []
    boot_delta_r2s = []
    
    np.random.seed(42)
    for _ in range(n_boot):
        # Sample CLUSTERS with replacement
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

# 5. Cluster-Level Paired Permutation Test
def cluster_paired_permutation(y_true, p0, p1, groups, n_perm=5000):
    unique_groups = np.unique(groups)
    d_rho_obs = pooled_d_rho
    d_r2_obs = pooled_d_r2
    
    count_rho = 0
    count_r2 = 0
    
    np.random.seed(42)
    for _ in range(n_perm):
        # Swap randomly at the cluster level
        swap_clusters = np.random.binomial(1, 0.5, size=len(unique_groups))
        swap_map = {g: bool(s) for g, s in zip(unique_groups, swap_clusters)}
        
        # Apply cluster-level swap to all observations
        swap = np.array([swap_map[g] for g in groups])
        
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

p_rho, p_r2 = cluster_paired_permutation(all_y_trues, all_m0_preds, all_m1_preds, all_groups)

print("--- Source-Expert Clustered Bootstrap CIs (Pooled ΔM1 - M0) ---")
print(f"Pooled Δρ 95% CI: [{ci_rho[0]:.3f}, {ci_rho[1]:.3f}]")
print(f"Pooled ΔR² 95% CI: [{ci_r2[0]:.3f}, {ci_r2[1]:.3f}]\n")

print("--- Cluster-Level Paired Permutation P-Values ---")
print(f"Pooled Δρ p-value: {p_rho:.5f}")
print(f"Pooled ΔR² p-value: {p_r2:.5f}")
