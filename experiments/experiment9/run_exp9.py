import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment4")))

from experiments.experiment4.data_loader import load_all
from experiments.experiment4.config import XGBOOST_PARAMS, LOCAL_FEATURES
from xgboost import XGBRegressor

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

def train_and_predict(X_train, y_train, X_test):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    return model.predict(X_test_scaled)

def main():
    # 1. Load basic Exp 4 data (target y, features Local-11)
    data = load_all()
    X_local_11 = data["X_unscaled"]
    y_target = data["y"]
    pair_i = data["pair_i"]
    pair_j = data["pair_j"]
    
    # 2. Load Usage and RW-L2 controls
    controls_path = os.path.join(os.path.dirname(__file__), "controls.json")
    with open(controls_path, "r") as f:
        controls = json.load(f)
    
    # Build dictionary for quick pair lookup
    controls_dict = {}
    for item in controls:
        a, b = item["pair"]
        key = (min(a, b), max(a, b))
        controls_dict[key] = item
        
    usage_sym = np.zeros(len(y_target), dtype=np.float32)
    rw_l2_sym = np.zeros(len(y_target), dtype=np.float32)
    
    for k in range(len(y_target)):
        key = (pair_i[k], pair_j[k])
        item = controls_dict[key]
        usage_sym[k] = item["usage_sum"]
        rw_l2_sym[k] = item["rw_l2_symmetric"]
        
    # 3. Load CV Splits from Exp 4
    exp4_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp4")
    cv_splits_path = os.path.join(exp4_dir, "cv_splits.json")
    with open(cv_splits_path, "r") as f:
        cv_splits = json.load(f)
        
    # We will evaluate over Partition 00 only to save time if we want, 
    # but let's do all 5 partitions to be robust, matching Exp 4.
    
    results_dict = {
        "Model 0": [],
        "Model 1": [],
        "Model 2": [],
        "Model 3": [],
        "Model 4": []
    }
    
    for part_idx, partition in enumerate(cv_splits):
        part_name = f"partition_{part_idx:02d}"
        print(f"\nProcessing {part_name}...")
        
        part_dir = os.path.join(exp4_dir, part_name)
        if not os.path.exists(part_dir):
            print(f"Skipping {part_name} (not found).")
            continue
            
        for fold_idx, fold in enumerate(partition["folds"]):
            fold_name = f"fold_{fold_idx:02d}"
            fold_dir = os.path.join(part_dir, fold_name)
            
            # Compute test/train indices from cv_splits
            train_experts_set = set(fold["train_experts"])
            test_experts_set = set(fold["test_experts"])
            
            train_idx = []
            test_idx = []
            for idx in range(len(pair_i)):
                if pair_i[idx] in train_experts_set and pair_j[idx] in train_experts_set:
                    train_idx.append(idx)
                elif pair_i[idx] in test_experts_set and pair_j[idx] in test_experts_set:
                    test_idx.append(idx)
                    
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
                
            # Get CARE Geometry
            # For Test pairs, Model B's prediction IS the geometry distance
            geom_test = np.load(os.path.join(fold_dir, "predictions_model_b.npy"))
            
            # For Train pairs, we compute distance from mds_coordinates.npy
            z_train = np.load(os.path.join(fold_dir, "mds_coordinates.npy"))
            with open(os.path.join(fold_dir, "train_experts.json"), "r") as f:
                train_experts_data = json.load(f)
            train_experts_list = train_experts_data["experts"] if isinstance(train_experts_data, dict) else train_experts_data
            expert_to_row = {int(eid): r for r, eid in enumerate(train_experts_list)}
            
            geom_train = np.zeros(len(train_idx), dtype=np.float32)
            for idx_in_train, global_idx in enumerate(train_idx):
                pi = pair_i[global_idx]
                pj = pair_j[global_idx]
                ri = expert_to_row[int(pi)]
                rj = expert_to_row[int(pj)]
                geom_train[idx_in_train] = float(np.linalg.norm(z_train[ri] - z_train[rj]))
                
            # Build feature matrices for this fold
            # Model 0: Local 11
            X0_train = X_local_11[train_idx]
            X0_test = X_local_11[test_idx]
            
            # Model 1: Usage + RW-L2
            X1_train = np.column_stack([usage_sym[train_idx], rw_l2_sym[train_idx]])
            X1_test = np.column_stack([usage_sym[test_idx], rw_l2_sym[test_idx]])
            
            # Model 3: CARE + Usage
            X3_train = np.column_stack([geom_train, usage_sym[train_idx]])
            X3_test = np.column_stack([geom_test, usage_sym[test_idx]])
            
            # Model 4: CARE + Usage + RW-L2
            X4_train = np.column_stack([geom_train, usage_sym[train_idx], rw_l2_sym[train_idx]])
            X4_test = np.column_stack([geom_test, usage_sym[test_idx], rw_l2_sym[test_idx]])
            
            y_train = y_target[train_idx]
            y_test = y_target[test_idx]
            
            # Evaluate Models
            pred0 = train_and_predict(X0_train, y_train, X0_test)
            pred1 = train_and_predict(X1_train, y_train, X1_test)
            pred2 = geom_test # Direct correlation
            pred3 = train_and_predict(X3_train, y_train, X3_test)
            pred4 = train_and_predict(X4_train, y_train, X4_test)
            
            # Calculate metrics
            def calc_metrics(pred, y_true):
                r2 = r2_score(y_true, pred) if len(pred) > 0 and len(np.unique(pred)) > 1 else 0
                rho, _ = stats.spearmanr(pred, y_true)
                return rho, r2
                
            from sklearn.metrics import r2_score
            
            for m_idx, pred in enumerate([pred0, pred1, pred2, pred3, pred4]):
                rho, r2 = calc_metrics(pred, y_test)
                results_dict[f"Model {m_idx}"].append({
                    "partition": part_idx,
                    "fold": fold_idx,
                    "rho": rho,
                    "r2": r2,
                    "pred": pred.tolist(),
                    "y_true": y_test.tolist(),
                    "source_expert_j": pair_j[test_idx].tolist() # For clustered bootstrap later
                })
                
    # Save results
    os.makedirs("experiments/experiment9/results", exist_ok=True)
    with open("experiments/experiment9/results/fold_results.json", "w") as f:
        json.dump(results_dict, f)
        
    print("\nExtraction and fold-wise training complete.")

if __name__ == "__main__":
    main()
