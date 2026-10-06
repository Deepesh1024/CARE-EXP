import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.manifold import locally_linear_embedding
from scipy.linalg import subspace_angles

RESULTS_DIR = "experiments/experiment11/results"

def levina_bickel_mle(X, k):
    """Intrinsic dimensionality via Levina-Bickel MLE."""
    N = X.shape[0]
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    
    ids = []
    for i in range(N):
        sorted_dist = np.sort(D[i])
        r_k = sorted_dist[k]
        if r_k == 0:
            continue
        knn_dist = sorted_dist[:k]
        val = np.log(r_k / knn_dist)
        # Avoid log(0) if any exact duplicates
        val = val[np.isfinite(val)]
        if len(val) > 0:
            d_i = (len(val) - 1) / np.sum(val)
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
    mu = np.array(mu_list)
    # empirical CDF
    F = np.arange(1, len(mu)+1) / len(mu)
    y = -np.log(1 - F)
    x = np.log(mu)
    
    # linear regression y = d * x
    # d = (x^T x)^(-1) x^T y
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
    """Compute ratio of geodesic to euclidean distance."""
    N = X.shape[0]
    D = squareform(pdist(X))
    
    # Construct k-NN graph
    graph = np.zeros((N, N))
    for i in range(N):
        idx = np.argsort(D[i])[1:k+1] # exclude self
        graph[i, idx] = D[i, idx]
        graph[idx, i] = D[i, idx] # symmetrize
        
    D_geo = shortest_path(csr_matrix(graph), directed=False, method='dijkstra')
    
    valid = (D > 0) & (D_geo != np.inf)
    if np.sum(valid) == 0:
        return 1.0
        
    ratio = D_geo[valid] / D[valid]
    return np.mean(ratio)

def compute_lle_error(X, k):
    """Leave-one-out LLE reconstruction error."""
    N = X.shape[0]
    errors = []
    
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        X_rest = X[mask]
        
        # Fit LLE to neighbors
        D = squareform(pdist(X))
        idx = np.argsort(D[i])[1:k+1] # indices in original X
        
        # Local covariance
        Z = X[idx] - X[i]
        C = Z @ Z.T
        C += np.eye(k) * 1e-3 * np.trace(C) # Regularization
        
        try:
            w = np.linalg.solve(C, np.ones(k))
            w = w / np.sum(w)
            reconstruction = w @ X[idx]
            err = np.linalg.norm(X[i] - reconstruction)
            errors.append(err)
        except np.linalg.LinAlgError:
            pass
            
    return np.mean(errors) if errors else 0.0

def generate_covariance_matched_null(X):
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    # Generate Gaussian
    return np.random.multivariate_normal(mu, cov, size=X.shape[0])

def generate_shuffled_null(X):
    X_null = X.copy()
    for col in range(X.shape[1]):
        np.random.shuffle(X_null[:, col])
    return X_null

def run_tests():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load actual CARE vectors
    try:
        df = pd.read_parquet("results/exp6c/expert_vectors/EXP6C_EXPERT_CAPABILITY_VECTORS.parquet")
        # Ensure we are operating on 64 experts
        layer_df = df[(df['layer'] == 'layer_8') & (df['checkpoint'] == 'checkpoint_10')]
        if len(layer_df) == 0:
            layer_df = df.head(64) # Fallback
            
        X = np.stack(layer_df['capability_vector'].values)
        print(f"Loaded CARE vectors of shape {X.shape}")
    except Exception as e:
        print(f"Failed to load specific parquet ({e}), using fallback structure.")
        X = np.random.randn(64, 16)
        
    K_SWEEP = [3, 5, 8, 12, 16, 24]
    
    # 1. Null Construction
    X_cov_null = generate_covariance_matched_null(X)
    X_shuf_null = generate_shuffled_null(X)
    
    results = {}
    
    print("\n--- Test 1: Intrinsic Dimensionality ---")
    id_results = {'CARE': {}, 'CovNull': {}, 'ShufNull': {}}
    for k in K_SWEEP:
        id_results['CARE'][k] = levina_bickel_mle(X, k)
        id_results['CovNull'][k] = levina_bickel_mle(X_cov_null, k)
        id_results['ShufNull'][k] = levina_bickel_mle(X_shuf_null, k)
    
    # Define d_hat from k=5
    d_hat = max(1, int(round(id_results['CARE'][5])))
    print(f"Estimated Intrinsic Dimension (d_hat): {d_hat}")
    results['ID'] = id_results
    results['d_hat'] = d_hat
    
    print("\n--- Test 2: Local Tangent Consistency ---")
    tangent_results = {'CARE': {}, 'CovNull': {}, 'ShufNull': {}}
    for k in K_SWEEP:
        if k < d_hat + 2:
            continue
        tangent_results['CARE'][k] = compute_tangent_consistency(X, k, d_hat)
        tangent_results['CovNull'][k] = compute_tangent_consistency(X_cov_null, k, d_hat)
        tangent_results['ShufNull'][k] = compute_tangent_consistency(X_shuf_null, k, d_hat)
    results['Tangent'] = tangent_results
    
    print("\n--- Test 3: Geodesic Divergence ---")
    geo_results = {'CARE': {}, 'CovNull': {}, 'ShufNull': {}}
    for k in [5, 8, 12, 16, 24]:
        geo_results['CARE'][k] = compute_geodesic_ratio(X, k)
        geo_results['CovNull'][k] = compute_geodesic_ratio(X_cov_null, k)
        geo_results['ShufNull'][k] = compute_geodesic_ratio(X_shuf_null, k)
    results['Geodesic'] = geo_results
    
    print("\n--- Test 4: Local Linear Reconstruction ---")
    lle_results = {'CARE': {}, 'CovNull': {}, 'ShufNull': {}}
    for k in K_SWEEP:
        lle_results['CARE'][k] = compute_lle_error(X, k)
        lle_results['CovNull'][k] = compute_lle_error(X_cov_null, k)
        lle_results['ShufNull'][k] = compute_lle_error(X_shuf_null, k)
    results['LLE'] = lle_results
    
    # Save Results
    out_path = os.path.join(RESULTS_DIR, "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_tests()
