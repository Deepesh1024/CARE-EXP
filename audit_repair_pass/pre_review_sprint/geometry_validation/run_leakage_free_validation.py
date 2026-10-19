"""
run_leakage_free_validation.py
==============================
Rigorous, Leakage-Free Empirical Evaluation of Pre-Merge Capability Features
Predicting Oracle Intervention Damage (Merge-KL) on OLMoE-1B-7B-0924.

Features Evaluated (Strictly Observational, Pre-Merge):
- Parameter Weight Distance (Frobenius / L2)
- Parameter Weight Cosine
- Token Activation Similarity
- Token Output Similarity
- Router Weight Routing Similarity
- Jaccard Expert Overlap
- Usage Frequency

Statistical Rigor:
- Expert-level cluster bootstrap (resampling experts, not pairs, to respect dyadic dependence)
- Non-parametric Spearman rho, Pearson r, Kendall tau
- Precision@Top-K (K in {10, 25, 50})
- Layer-by-layer breakdown: First, Middle, Last layers (N = 8,064 pairs per layer)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP1_PATH = os.path.join(PROJECT_ROOT, "results", "exp1", "output.json")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "LEAKAGE_FREE_RESULTS.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "LEAKAGE_FREE_RESULTS.json")

def compute_cluster_bootstrap_ci(df, feature_col, target_col, n_boot=1000, seed=42):
    """Resamples experts with replacement to account for dyadic dependence."""
    rng = np.random.RandomState(seed)
    experts = np.unique(np.concatenate([df['Expert_A'].values, df['Expert_B'].values]))
    n_experts = len(experts)
    
    boot_rhos = []
    for _ in range(n_boot):
        boot_experts = set(rng.choice(experts, size=n_experts, replace=True))
        # Keep pairs where both experts are in the bootstrap sample
        mask = df['Expert_A'].isin(boot_experts) & df['Expert_B'].isin(boot_experts)
        boot_sub = df[mask]
        if len(boot_sub) > 20:
            rho, _ = spearmanr(boot_sub[feature_col], boot_sub[target_col])
            if not np.isnan(rho):
                boot_rhos.append(rho)
                
    if not boot_rhos:
        return np.nan, np.nan, np.nan
    
    ci_low = np.percentile(boot_rhos, 2.5)
    ci_high = np.percentile(boot_rhos, 97.5)
    se = np.std(boot_rhos)
    return se, ci_low, ci_high

def compute_precision_at_k(df, feature_col, target_col, k_vals=[10, 25, 50], ascending=True):
    """
    Computes Precision@Top-K: fraction of top-K predicted candidates
    that fall within the true top-K lowest damage candidates.
    """
    precisions = {}
    n = len(df)
    target_sorted = df.sort_values(by=target_col, ascending=True).index
    feature_sorted = df.sort_values(by=feature_col, ascending=ascending).index
    
    for k in k_vals:
        if k > n:
            precisions[f"P@{k}"] = np.nan
            continue
        top_k_true = set(target_sorted[:k])
        top_k_pred = set(feature_sorted[:k])
        overlap = len(top_k_true.intersection(top_k_pred))
        precisions[f"P@{k}"] = overlap / float(k)
        
    return precisions

def main():
    print("=" * 70)
    print("CARE-COM PRE-REVIEW SPRINT: LEAKAGE-FREE GEOMETRY VALIDATION")
    print("=" * 70)
    
    print(f"Loading raw observational records from: {EXP1_PATH}")
    with open(EXP1_PATH, "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data['results'])
    print(f"Total raw pairs loaded: {len(df)}")
    
    features = [
        ('Weight_Distance', True),        # Smaller distance -> smaller damage
        ('Weight_Cosine', False),         # Larger cosine -> smaller damage
        ('Activation_Similarity', False), # Larger similarity -> smaller damage
        ('Output_Similarity', False),     # Larger similarity -> smaller damage
        ('Routing_Similarity', False),    # Larger similarity -> smaller damage
        ('Usage_Frequency', False),
        ('Jaccard_Overlap', False)
    ]
    
    results = []
    
    layers = ['all', 'first', 'middle', 'last']
    
    for layer in layers:
        layer_df = df if layer == 'all' else df[df['Layer'] == layer].copy()
        n_obs = len(layer_df)
        print(f"\nAnalyzing Layer: {layer.upper()} (N = {n_obs} candidate pairs)")
        
        for feat_name, asc in features:
            if feat_name not in layer_df.columns:
                continue
                
            x = layer_df[feat_name].values
            y = layer_df['Oracle_KL'].values
            
            # Basic correlations
            rho, p_val = spearmanr(x, y)
            r_pearson, p_pearson = pearsonr(x, y)
            tau, p_tau = kendalltau(x, y)
            
            # Cluster-robust standard error via expert bootstrap
            se, ci_low, ci_high = compute_cluster_bootstrap_ci(layer_df, feat_name, 'Oracle_KL', n_boot=500)
            
            # Precision@K
            precs = compute_precision_at_k(layer_df, feat_name, 'Oracle_KL', k_vals=[10, 25, 50], ascending=asc)
            
            row = {
                'layer': layer,
                'feature': feat_name,
                'n_pairs': n_obs,
                'spearman_rho': float(rho),
                'spearman_p': float(p_val),
                'cluster_se': float(se),
                'ci_95_low': float(ci_low),
                'ci_95_high': float(ci_high),
                'pearson_r': float(r_pearson),
                'kendall_tau': float(tau),
                'prec_at_10': precs.get('P@10'),
                'prec_at_25': precs.get('P@25'),
                'prec_at_50': precs.get('P@50')
            }
            results.append(row)
            
            direction = "(-) protective" if rho < 0 else "(+) damaging"
            print(f"  {feat_name:22s} | rho = {rho:+.4f} (SE = {se:.4f}, 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]) | {direction}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved CSV report to: {OUTPUT_CSV}")
    print(f"Saved JSON report to: {OUTPUT_JSON}")
    print("=" * 70)

if __name__ == "__main__":
    main()
