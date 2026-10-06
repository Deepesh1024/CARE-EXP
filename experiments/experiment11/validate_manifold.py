import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from scipy.optimize import nnls

RESULTS_DIR = "experiments/experiment11/results"
N_NULL_SAMPLES = 50

def levina_bickel_mle(X, k):
    """Intrinsic dimensionality via Levina-Bickel MLE."""
    N = X.shape[0]
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    
    ids = []
    for i in range(N):
        sorted_dist = np.sort(D[i])
        r_k = sorted_dist[k-1] # distance to k-th nearest neighbor
        if r_k <= 0:
            continue
        knn_dist = sorted_dist[:k-1]
        val = np.log(r_k / knn_dist)
        val = val[np.isfinite(val)]
        if len(val) > 0:
            d_i = len(val) / np.sum(val)
            ids.append(d_i)
    return np.mean(ids) if ids else 0

def twonn_estimator(X):
    """Intrinsic dimensionality via TwoNN."""
    N = X.shape[0]
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    
    mu_list = []
    for i in range(N):
        sorted_dist = np.sort(D[i])
        r1 = sorted_dist[0]
        r2 = sorted_dist[1]
        if r1 > 0:
            mu_list.append(r2 / r1)
            
    if not mu_list:
        return 0
    mu = np.sort(np.array(mu_list))
    # Drop last point to avoid log(1-F) -> infinity
    mu = mu[:-1]
    
    F = np.arange(1, len(mu)+1) / (len(mu)+1)
    y = -np.log(1 - F)
    x = np.log(mu)
    
    x = x[:, np.newaxis]
    d = np.linalg.lstsq(x, y, rcond=None)[0][0]
    return d

def compute_tangent_consistency(X, k, d_hat):
    """Compute local tangent spaces and measure consistency."""
    N = X.shape[0]
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    
    U = []
    for i in range(N):
        idx = np.argsort(D[i])[:k]
        neighbors = X[idx]
        pca = PCA(n_components=d_hat)
        pca.fit(neighbors)
        U.append(pca.components_.T) # shape: (K, d_hat)
        
    angles_list = []
    for i in range(N):
        idx = np.argsort(D[i])[:k]
        for j in idx:
            angles = subspace_angles(U[i], U[j])
            angles_list.extend(angles)
            
    return np.mean(angles_list)

def compute_geodesic_ratio(X, k):
    """Compute ratio of geodesic to euclidean distance, returning connectivity rate."""
    N = X.shape[0]
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    
    graph = np.zeros((N, N))
    for i in range(N):
        idx = np.argsort(D[i])[:k]
        graph[i, idx] = D[i, idx]
        graph[idx, i] = D[i, idx] # symmetrize
        
    n_components, labels = connected_components(csgraph=csr_matrix(graph), directed=False, return_labels=True)
    
    D_geo = shortest_path(csr_matrix(graph), directed=False, method='auto')
    
    valid = (D > 0) & (D_geo != np.inf)
    conn_rate = np.sum(valid) / (N * (N-1))
    
    if np.sum(valid) == 0:
        return 1.0, conn_rate
        
    ratio = D_geo[valid] / D[valid]
    return np.mean(ratio), conn_rate

def compute_lle_error(X, k):
    """Leave-one-out LLE reconstruction error with strictly non-negative weights."""
    N = X.shape[0]
    errors = []
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    
    for i in range(N):
        idx = np.argsort(D[i])[:k]
        Z = X[idx].T # (ambient_dim, k)
        target = X[i] # (ambient_dim,)
        
        w, _ = nnls(Z, target)
        if np.sum(w) > 0:
            w = w / np.sum(w)
        else:
            w = np.ones(k) / k
            
        recon = Z @ w
        err = np.linalg.norm(target - recon)
        errors.append(err)
            
    return np.mean(errors)

def compute_bootstrap_stability(X, k, d_hat, n_boot=20):
    """Representation stability via Gaussian perturbation."""
    N, K = X.shape
    jaccards = []
    angles = []
    
    D_base = squareform(pdist(X))
    np.fill_diagonal(D_base, np.inf)
    
    for _ in range(n_boot):
        # Small perturbation equivalent to probe-subsampling variance
        noise_scale = 0.05
        X_boot = X + noise_scale * np.random.randn(N, K)
        X_boot = X_boot / np.linalg.norm(X_boot, axis=1, keepdims=True)
        
        D_boot = squareform(pdist(X_boot))
        np.fill_diagonal(D_boot, np.inf)
        
        for i in range(N):
            nn_base = set(np.argsort(D_base[i])[:k])
            nn_boot = set(np.argsort(D_boot[i])[:k])
            jacc = len(nn_base.intersection(nn_boot)) / len(nn_base.union(nn_boot))
            jaccards.append(jacc)
            
            if k >= d_hat + 2:
                pca_base = PCA(n_components=d_hat).fit(X[list(nn_base)])
                pca_boot = PCA(n_components=d_hat).fit(X_boot[list(nn_boot)])
                ang = subspace_angles(pca_base.components_.T, pca_boot.components_.T)
                angles.extend(ang)
                
    return np.mean(jaccards), np.mean(angles) if angles else np.nan

def generate_covariance_matched_null(X):
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    return np.random.multivariate_normal(mu, cov, size=X.shape[0])

def generate_shuffled_null(X):
    X_null = X.copy()
    for col in range(X.shape[1]):
        np.random.shuffle(X_null[:, col])
    return X_null

def generate_positive_control(X):
    """Generate a 2D Swiss Roll embedded in X's ambient dimension, matched to its covariance scale."""
    N, K = X.shape
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(N))
    x = t * np.cos(t)
    y = 21 * np.random.rand(N)
    z = t * np.sin(t)
    
    roll = np.column_stack((x, y, z))
    Q, _ = np.linalg.qr(np.random.randn(K, K))
    roll_embedded = np.zeros((N, K))
    roll_embedded[:, :3] = roll
    roll_embedded = roll_embedded @ Q.T
    
    var_X = np.trace(np.cov(X, rowvar=False))
    var_roll = np.trace(np.cov(roll_embedded, rowvar=False))
    roll_embedded *= np.sqrt(var_X / var_roll)
    roll_embedded += 0.05 * np.random.randn(N, K) * np.std(roll_embedded)
    return roll_embedded

def compute_empirical_p(val, null_vals, tail="right"):
    null_vals = np.array(null_vals)
    if tail == "right":
        return (np.sum(null_vals >= val) + 1) / (len(null_vals) + 1)
    elif tail == "left":
        return (np.sum(null_vals <= val) + 1) / (len(null_vals) + 1)
    return np.nan

def run_tests():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load actual CARE vectors
    df = pd.read_parquet("results/exp6c/expert_vectors/EXP6C_EXPERT_CAPABILITY_VECTORS.parquet")
    layer_df = df[(df['layer_idx'] == 8) & (df['checkpoint'] == 'checkpoint_10')]
    if len(layer_df) != 64:
        raise RuntimeError("Failed to correctly load 64 experts for the target layer. Data missing or corrupted.")
        
    X = np.stack(layer_df['C_hat'].values)
    print(f"Loaded exact CARE C_hat vectors of shape {X.shape}")
    
    K_SWEEP = [3, 5, 8, 12, 16]
    results = {'CARE': {}, 'CovNull': {}, 'ShufNull': {}, 'PosCtrl': {}}
    
    # Initialize metric storage
    for subset in ['CARE', 'PosCtrl']:
        results[subset] = {'ID': {}, 'Tangent': {}, 'Geodesic': {}, 'ConnRate': {}, 'LLE': {}, 'BootJaccard': {}, 'BootAngle': {}}
        
    # Null distributions
    results['CovNull'] = {'ID': {k: [] for k in K_SWEEP}, 'Tangent': {k: [] for k in K_SWEEP}, 
                          'Geodesic': {k: [] for k in K_SWEEP}, 'ConnRate': {k: [] for k in K_SWEEP}, 
                          'LLE': {k: [] for k in K_SWEEP}}
                          
    results['ShufNull'] = {'ID': {k: [] for k in K_SWEEP}, 'Tangent': {k: [] for k in K_SWEEP}, 
                           'Geodesic': {k: [] for k in K_SWEEP}, 'ConnRate': {k: [] for k in K_SWEEP}, 
                           'LLE': {k: [] for k in K_SWEEP}}
    
    X_pos_ctrl = generate_positive_control(X)
    
    print("\n[Phase 1] Computing CARE and Positive Control Metrics...")
    # Base ID
    for k in K_SWEEP:
        results['CARE']['ID'][k] = levina_bickel_mle(X, k)
        results['PosCtrl']['ID'][k] = levina_bickel_mle(X_pos_ctrl, k)
        
    d_hat_care = max(1, int(round(results['CARE']['ID'][5])))
    d_hat_pos = max(1, int(round(results['PosCtrl']['ID'][5])))
    
    print(f"  CARE estimated intrinsic dimension: {d_hat_care}")
    print(f"  PosCtrl estimated intrinsic dimension: {d_hat_pos}")
    
    for k in K_SWEEP:
        # Tangent
        results['CARE']['Tangent'][k] = compute_tangent_consistency(X, k, d_hat_care) if k >= d_hat_care + 2 else np.nan
        results['PosCtrl']['Tangent'][k] = compute_tangent_consistency(X_pos_ctrl, k, d_hat_pos) if k >= d_hat_pos + 2 else np.nan
        
        # Geodesic
        geo_c, conn_c = compute_geodesic_ratio(X, k)
        geo_p, conn_p = compute_geodesic_ratio(X_pos_ctrl, k)
        results['CARE']['Geodesic'][k] = geo_c
        results['CARE']['ConnRate'][k] = conn_c
        results['PosCtrl']['Geodesic'][k] = geo_p
        results['PosCtrl']['ConnRate'][k] = conn_p
        
        # LLE
        results['CARE']['LLE'][k] = compute_lle_error(X, k)
        results['PosCtrl']['LLE'][k] = compute_lle_error(X_pos_ctrl, k)
        
        # Bootstrap Stability
        jacc_c, ang_c = compute_bootstrap_stability(X, k, d_hat_care)
        results['CARE']['BootJaccard'][k] = jacc_c
        results['CARE']['BootAngle'][k] = ang_c
        
    print(f"\n[Phase 2] Generating {N_NULL_SAMPLES} Null Realizations...")
    for step in range(N_NULL_SAMPLES):
        X_cov = generate_covariance_matched_null(X)
        X_shuf = generate_shuffled_null(X)
        
        for k in K_SWEEP:
            # ID
            results['CovNull']['ID'][k].append(levina_bickel_mle(X_cov, k))
            results['ShufNull']['ID'][k].append(levina_bickel_mle(X_shuf, k))
            
            # Tangent
            results['CovNull']['Tangent'][k].append(compute_tangent_consistency(X_cov, k, d_hat_care) if k >= d_hat_care + 2 else np.nan)
            results['ShufNull']['Tangent'][k].append(compute_tangent_consistency(X_shuf, k, d_hat_care) if k >= d_hat_care + 2 else np.nan)
            
            # Geodesic
            g_cov, c_cov = compute_geodesic_ratio(X_cov, k)
            g_shuf, c_shuf = compute_geodesic_ratio(X_shuf, k)
            results['CovNull']['Geodesic'][k].append(g_cov)
            results['CovNull']['ConnRate'][k].append(c_cov)
            results['ShufNull']['Geodesic'][k].append(g_shuf)
            results['ShufNull']['ConnRate'][k].append(c_shuf)
            
            # LLE
            results['CovNull']['LLE'][k].append(compute_lle_error(X_cov, k))
            results['ShufNull']['LLE'][k].append(compute_lle_error(X_shuf, k))
            
    print("\n[Phase 3] Extracting Statistics and P-Values...")
    final_report = {}
    for k in K_SWEEP:
        final_report[k] = {
            'CARE_ID': results['CARE']['ID'][k],
            'CovNull_ID_mean': np.mean(results['CovNull']['ID'][k]),
            'CovNull_ID_pval': compute_empirical_p(results['CARE']['ID'][k], results['CovNull']['ID'][k], tail="left"),
            
            'CARE_Tangent': results['CARE']['Tangent'][k],
            'CovNull_Tangent_mean': np.mean(results['CovNull']['Tangent'][k]),
            'CovNull_Tangent_pval': compute_empirical_p(results['CARE']['Tangent'][k], results['CovNull']['Tangent'][k], tail="left"),
            
            'CARE_Geodesic': results['CARE']['Geodesic'][k],
            'CARE_ConnRate': results['CARE']['ConnRate'][k],
            'CovNull_Geodesic_mean': np.mean(results['CovNull']['Geodesic'][k]),
            'CovNull_Geodesic_pval': compute_empirical_p(results['CARE']['Geodesic'][k], results['CovNull']['Geodesic'][k], tail="right"),
            
            'CARE_LLE': results['CARE']['LLE'][k],
            'CovNull_LLE_mean': np.mean(results['CovNull']['LLE'][k]),
            'CovNull_LLE_pval': compute_empirical_p(results['CARE']['LLE'][k], results['CovNull']['LLE'][k], tail="left"),
            
            'PosCtrl_LLE': results['PosCtrl']['LLE'][k],
            'CARE_BootJaccard': results['CARE']['BootJaccard'][k]
        }
        
    out_path = os.path.join(RESULTS_DIR, "validation_results_v2.json")
    with open(out_path, "w") as f:
        json.dump({'Raw': results, 'Summary': final_report}, f, indent=4)
        
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_tests()
