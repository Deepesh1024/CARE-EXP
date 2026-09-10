"""
EXPERIMENT 7A - PHASE 3: STRATIFIED NEURON SAMPLING
=====================================================
Reads T_K sensitivity scores, partitions into 5 quantiles, 
and samples 10 neurons from each (total 50).
"""

import os
import json
import numpy as np
import pandas as pd
import random

from config import (
    N_EXPERTS, INTERMEDIATE_SIZE, RANDOM_SEED, TARGET_LAYER_IDX,
    SENSITIVITY_DIR, INTERVENTION_DIR, EXP7A_RESULTS_DIR,
    ensure_dirs, mark_task, is_task_completed
)

def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def run_sampling():
    task_id = "phase3_sampling"
    if is_task_completed(task_id):
        print("[Phase 3] Sampling already completed. Skipping.")
        return

    ensure_dirs()
    set_seed()

    tk_path = os.path.join(SENSITIVITY_DIR, "T_K.npy")
    if not os.path.exists(tk_path):
        raise FileNotFoundError(f"Missing {tk_path}. Run phase2 first.")

    t_k_scores = np.load(tk_path)
    
    # Flatten the (N_EXPERTS, INTERMEDIATE_SIZE) array to rank globally
    records = []
    for e in range(N_EXPERTS):
        for n in range(INTERMEDIATE_SIZE):
            records.append({
                "layer": TARGET_LAYER_IDX,
                "expert_idx": e,
                "neuron_idx": n,
                "T_K": float(t_k_scores[e, n])
            })
            
    df = pd.DataFrame(records)
    
    # Exclude neurons from Experiment 7A to ensure zero overlap
    exp7a_csv_path = os.path.join(EXP7A_RESULTS_DIR, "intervention", "sampled_neurons.csv")
    if os.path.exists(exp7a_csv_path):
        exp7a_sampled = pd.read_csv(exp7a_csv_path)
        exp7a_set = set(zip(exp7a_sampled['expert_idx'], exp7a_sampled['neuron_idx']))
        print(f"[Phase 3] Found {len(exp7a_set)} neurons from 7A. Excluding them from the new sample.")
        df = df[~df.apply(lambda row: (int(row['expert_idx']), int(row['neuron_idx'])) in exp7a_set, axis=1)]
    else:
        raise FileNotFoundError(f"Missing {exp7a_csv_path}. Cannot guarantee zero overlap without it.")
        
    # Sort by T_K ascending
    df = df.sort_values(by="T_K").reset_index(drop=True)
    
    # Divide into 5 fixed quantile strata
    # qcut ensures roughly equal sized bins
    df['stratum'] = pd.qcut(df.index, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    sampled_neurons = []
    print("\n[Phase 3] Taylor Sensitivity Strata Boundaries:")
    for stratum in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        stratum_df = df[df['stratum'] == stratum]
        min_tk = stratum_df['T_K'].min()
        max_tk = stratum_df['T_K'].max()
        print(f"  {stratum}: Min T_K = {min_tk:.6f}, Max T_K = {max_tk:.6f}, Count = {len(stratum_df)}")
        # Sample 10 neurons randomly
        sampled = stratum_df.sample(n=10, random_state=RANDOM_SEED)
        sampled_neurons.extend(sampled.to_dict('records'))
        
    sampled_df = pd.DataFrame(sampled_neurons)
    
    # Save sampled list
    out_csv = os.path.join(INTERVENTION_DIR, "sample_manifest.csv")
    sampled_df.to_csv(out_csv, index=False)
    
    # Save structured json manifest
    manifest = {
        "seed": RANDOM_SEED,
        "total_population": len(df),
        "total_sampled": len(sampled_df),
        "strata_counts": sampled_df['stratum'].value_counts().to_dict(),
        "neurons": sampled_neurons
    }
    with open(os.path.join(INTERVENTION_DIR, "sampled_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"[Phase 3] Sampled {len(sampled_df)} neurons. Saved to {out_csv}")
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_sampling()
