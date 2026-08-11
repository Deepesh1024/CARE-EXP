import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import phase3_cross_validation as p3

def audit_fold_q():
    layer = "middle"
    q = 4
    
    # Load D
    df = pd.read_csv(f"../../results/exp3b/oracle_distance_matrix_{layer}.csv")
    D = df.values.astype(float)
    n = D.shape[0]
    
    rep_seed = p3.RANDOM_SEED
    kf = KFold(n_splits=p3.N_FOLDS, shuffle=True, random_state=rep_seed)
    
    # Get just the first fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n))):
        break
        
    print(f"Test indices ({len(test_idx)}):", test_idx)
    
    fold_seed = rep_seed * 100 + fold * 10 + q
    
    D_train = D[np.ix_(train_idx, train_idx)]
    Z_train = p3.run_smacof_train(D_train, q, fold_seed)
    
    Z_test = np.zeros((len(test_idx), q))
    for t, test_exp in enumerate(test_idx):
        d_to_train = D[test_exp, train_idx]
        Z_test[t] = p3.embed_single_test_expert(
            Z_train, d_to_train, q,
            seed=(fold_seed * 10000 + t) % (2**32 - 1)
        )
        
    # Test->Test
    test_test_embed_dists = []
    test_test_oracle_dists = []

    for i in range(len(test_idx)):
        for j in range(i + 1, len(test_idx)):
            embed_d = np.linalg.norm(Z_test[i] - Z_test[j])
            oracle_d = D[test_idx[i], test_idx[j]]
            test_test_embed_dists.append(embed_d)
            test_test_oracle_dists.append(oracle_d)

    test_test_embed_dists = np.array(test_test_embed_dists)
    test_test_oracle_dists = np.array(test_test_oracle_dists)

    print("\nTest-Test Distance Audit:")
    print("Oracle mean:", np.mean(test_test_oracle_dists), "std:", np.std(test_test_oracle_dists))
    print("Embed mean:", np.mean(test_test_embed_dists), "std:", np.std(test_test_embed_dists))
    
    rmse = np.sqrt(np.mean((test_test_embed_dists - test_test_oracle_dists)**2))
    pearson = pearsonr(test_test_oracle_dists, test_test_embed_dists).statistic
    spearman = spearmanr(test_test_oracle_dists, test_test_embed_dists).statistic
    
    print(f"RMSE: {rmse:.6f}")
    print(f"Pearson r: {pearson:.4f}")
    print(f"Spearman rho: {spearman:.4f}")
    
    print("\nOracle distances (first 10):", test_test_oracle_dists[:10])
    print("Embed distances (first 10):", test_test_embed_dists[:10])

if __name__ == "__main__":
    audit_fold_q()
