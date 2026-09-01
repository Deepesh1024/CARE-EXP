import sys, os, time
import numpy as np
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, '/Users/user/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4')
from data_loader import load_all
from cv_splits import get_fold_data, generate_and_freeze_splits
from config import Q, SMACOF_N_INIT, SMACOF_MAX_ITER, SMACOF_EPS, SMACOF_METRIC, OOS_N_RESTARTS
from mds_embedding import embed_test_expert
from model_a import train_model_a
import json

def run_audit():
    data = load_all()
    splits = generate_and_freeze_splits()
    
    D_oracle = data["D_oracle"]
    X_unscaled = data["X_unscaled"]
    y = data["y"]
    pair_i = data["pair_i"]
    pair_j = data["pair_j"]

    fold_data = get_fold_data(splits, 0, 0, pair_i, pair_j, X_unscaled, y)
    train_experts = fold_data["train_experts"]
    test_experts = fold_data["test_experts"]
    X_train = fold_data["X_train"]
    y_train = fold_data["y_train"]
    train_idx = np.array(train_experts, dtype=np.int32)
    test_idx = np.array(test_experts, dtype=np.int32)
    D_train = D_oracle[np.ix_(train_idx, train_idx)].astype(np.float64)

    from sklearn.manifold import MDS
    
    audit_data = {}
    audit_data["SMACOF_restarts"] = []
    
    print("Fold P0F0 Audit:")
    seed = 1001 * 1000 + 0
    best_Z = None
    best_stress = np.inf
    
    start_time = time.time()
    for init_idx in range(SMACOF_N_INIT):
        init_seed = (seed * 10000 + init_idx) % (2**31 - 1)
        
        mds = MDS(
            n_components=Q,
            metric=SMACOF_METRIC,
            dissimilarity="precomputed",
            max_iter=SMACOF_MAX_ITER,
            n_init=1,
            eps=SMACOF_EPS,
            random_state=init_seed,
            n_jobs=1,
            normalized_stress=False
        )
        t0 = time.time()
        Z = mds.fit_transform(D_train)
        t1 = time.time()
        
        # Calculate initial stress (by fitting with max_iter=0, but MDS doesn't allow 0 easily)
        # We can just say initial stress is unknown via MDS, but we have final stress and iter.
        audit_data["SMACOF_restarts"].append({
            "restart": init_idx + 1,
            "iterations": int(mds.n_iter_),
            "final_stress": float(mds.stress_),
            "time_sec": float(t1-t0)
        })
        print(f"  Restart {init_idx+1}: Iterations: {mds.n_iter_}, Stress: {mds.stress_}, Time: {t1-t0:.4f}s")
        
        if mds.stress_ < best_stress:
            best_stress = mds.stress_
            best_Z = Z

    total_mds = time.time() - start_time
    audit_data["total_smacof_time"] = total_mds
    
    # OOS Test experts
    t0 = time.time()
    for t, test_exp in enumerate(test_idx):
        d_to_train = D_oracle[test_exp, train_idx].astype(np.float64)
        embed_seed = (seed * 10000 + t) % (2**31 - 1)
        z_t = embed_test_expert(best_Z, d_to_train, Q, embed_seed)
    total_oos = time.time() - t0
    audit_data["total_oos_time"] = total_oos
    print(f"Total OOS embedding time: {total_oos:.4f}s")
    
    # XGBoost time
    t0 = time.time()
    train_model_a(X_train, y_train)
    total_xgboost = time.time() - t0
    audit_data["total_xgboost_time"] = total_xgboost
    print(f"Total XGBoost time: {total_xgboost:.4f}s")
    
    with open('/Users/user/Desktop/CARE-MoE/Experiments-V3/results/exp4/runtime_audit_temp.json', 'w') as f:
        json.dump(audit_data, f, indent=2)

if __name__ == "__main__":
    run_audit()
