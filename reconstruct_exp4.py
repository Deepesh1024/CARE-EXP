import os
import glob
import json
import numpy as np
import pandas as pd

results_dir = "results/exp4"
partitions = glob.glob(os.path.join(results_dir, "partition_*"))

all_pairs = []

for part in partitions:
    part_name = os.path.basename(part)
    layer = int(part_name.split("_")[1]) if "_" in part_name else part_name
    
    folds = glob.glob(os.path.join(part, "fold_*"))
    for fold in folds:
        fold_name = os.path.basename(fold)
        
        pair_indices_path = os.path.join(fold, "pair_indices.json")
        oracle_path = os.path.join(fold, "oracle_targets.npy")
        pred_a_path = os.path.join(fold, "predictions_model_a.npy")
        pred_b_path = os.path.join(fold, "predictions_model_b.npy")
        pred_c_path = os.path.join(fold, "predictions_model_c.npy")
        
        if not all(os.path.exists(p) for p in [pair_indices_path, oracle_path, pred_a_path, pred_b_path]):
            continue
            
        with open(pair_indices_path, "r") as f:
            pair_indices = json.load(f)
            
        oracle = np.load(oracle_path)
        pred_a = np.load(pred_a_path)
        pred_b = np.load(pred_b_path)
        
        has_c = os.path.exists(pred_c_path)
        pred_c = np.load(pred_c_path) if has_c else [None]*len(oracle)
        
        for i, pair in enumerate(pair_indices):
            all_pairs.append({
                "pair_id": f"{part_name}_{fold_name}_{pair[0]}_{pair[1]}",
                "layer": layer,
                "validation_fold": fold_name,
                "expert_1": pair[0],
                "expert_2": pair[1],
                "oracle_kl": oracle[i],
                "local_baseline_score": pred_a[i],
                "capability_distance": pred_b[i],
                "combined_score": pred_c[i] if has_c else None
            })

df = pd.DataFrame(all_pairs)
df.to_csv("experiment4_pair_level.csv", index=False)
print(f"Saved {len(df)} pairs to experiment4_pair_level.csv")
