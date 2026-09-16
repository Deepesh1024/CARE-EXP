import os
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import warnings

RESULTS_DIR_7C = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7c'))

def run_residual_analysis():
    print("=" * 70)
    print("EXPERIMENT 7C — PHASE 8: RESIDUAL ANALYSIS")
    print("=" * 70)

    # 1. Load actual merges from 7B
    merges_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7b', 'merges', 'actual_merge_results.csv')
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Missing 7B actual merge results: {merges_path}")
        
    merges_df = pd.read_csv(merges_path)
    
    # Residual definition: R_ij = D_actual(i,j) - D_pred(i,j)
    # The 'Error_KL' column from 7B is exactly D_actual_KL - D_pred
    merges_df['residual'] = merges_df['D_actual_KL'] - merges_df['D_pred']
    
    # 2. Load cloud analysis results
    c_path = os.path.join(RESULTS_DIR_7C, "revised", "analysis", "cloud_analysis_results.csv")
    if not os.path.exists(c_path):
        raise FileNotFoundError(f"Missing 7C cloud analysis results: {c_path}")
        
    cloud_df = pd.read_csv(c_path)
    
    # 3. Join strictly by pair_id to prevent row misalignment
    joined_df = pd.merge(
        merges_df[['pair_id', 'expert_i', 'expert_j', 'D_pred', 'D_actual_KL', 'residual']],
        cloud_df[['pair_id', 'mutual_count', 'mutual_frac', 'C_i_to_j', 'C_j_to_i', 'C_mutual']],
        on='pair_id',
        how='inner'
    )
    
    # Verify all 18 pairs are present
    if len(joined_df) != 18:
        warnings.warn(f"Expected 18 pairs, found {len(joined_df)} after join. Check exclusions.")
        
    if len(joined_df) == 0:
        print("No valid pairs to analyze. Exiting.")
        return

    # 4. Primary Statistical Test
    C_mutual = joined_df['C_mutual'].values
    residual = joined_df['residual'].values
    
    rho, p_value = spearmanr(C_mutual, residual)
    
    # Calculate simple bootstrap CI for Spearman rho
    n_iterations = 1000
    n_size = len(C_mutual)
    boot_rhos = []
    
    # Set seed for reproducibility
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        boot_rho, _ = spearmanr(C_mutual[indices], residual[indices])
        if not np.isnan(boot_rho):
            boot_rhos.append(boot_rho)
            
    ci_lower = np.percentile(boot_rhos, 2.5) if boot_rhos else np.nan
    ci_upper = np.percentile(boot_rhos, 97.5) if boot_rhos else np.nan

    print("\n--- PRIMARY STATISTICAL ANALYSIS ---")
    print(f"Sample Size (N)   = {n_size} candidate pairs")
    print(f"Spearman rho      = {rho:+.4f}")
    print(f"p-value           = {p_value:.4e}")
    print(f"95% Bootstrap CI  = [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print("\nNote: N=18 candidate pairs. This analysis is hypothesis-generating.")
    
    # 5. Save joined table
    out_dir = os.path.join(RESULTS_DIR_7C, "revised", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "7c_final_residual_analysis.csv")
    joined_df.to_csv(out_path, index=False)
    
    print(f"\nFinal joined analysis table saved to: {out_path}")
    
    # 6. Secondary Exploratory Analysis
    print("\n--- SECONDARY EXPLORATORY ANALYSIS ---")
    rho_i_j, p_i_j = spearmanr(joined_df['C_i_to_j'].values, residual)
    rho_j_i, p_j_i = spearmanr(joined_df['C_j_to_i'].values, residual)
    
    print(f"C_i_to_j vs residual: rho={rho_i_j:+.4f}, p={p_i_j:.4f}")
    print(f"C_j_to_i vs residual: rho={rho_j_i:+.4f}, p={p_j_i:.4f}")
    print("These results are exploratory and do not override the primary statistic.")

if __name__ == "__main__":
    run_residual_analysis()
