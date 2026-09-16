import os
import json
import numpy as np
import pandas as pd

def audit_mutual_support():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_7c_dir = os.path.join(base_dir, 'results', 'exp7c')
    results_7b_dir = os.path.join(base_dir, 'results', 'exp7b')
    
    memmap_path = os.path.join(results_7c_dir, 'activations', 'expert_signatures.npy')
    meta_path = os.path.join(results_7c_dir, 'activations', 'expert_signatures_meta.json')
    merges_path = os.path.join(results_7b_dir, 'merges', 'actual_merge_results.csv')
    
    if not os.path.exists(memmap_path):
        print(f"Error: {memmap_path} not found. Must run this on the VM.")
        return
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    shape = tuple(meta['shape'])
    dtype = np.float16 if meta['dtype'] == 'float16' else np.float32
    expert_mapping = meta['expert_mapping']
    num_tokens = meta['num_tokens']
    
    print(f"Loading memmap: {shape}, dtype={dtype}")
    activations = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
    
    cand_df = pd.read_csv(merges_path)
    pairs = cand_df[['pair_id', 'expert_i', 'expert_j']].to_dict('records')
    
    print("\n==============================================================")
    print("PHASE 1 — AUDITING MUTUAL ACTIVATION SUPPORT")
    print("==============================================================\n")
    print(f"{'Pair':<10} {'Exp_i':<7} {'Exp_j':<7} {'Total_Tokens':<15} {'Mutual_Tokens':<15} {'Mutual_Frac':<15} {'Status'}")
    print("-" * 80)
    
    total_sufficient = 0
    
    # We consider support sufficient if we have at least 100 mutual tokens for a stable cosine similarity.
    # We'll see empirically what the distribution looks like.
    THRESHOLD = 50
    
    for pair in pairs:
        pid = pair['pair_id']
        ei = str(pair['expert_i'])
        ej = str(pair['expert_j'])
        
        idx_i = expert_mapping[ei]
        idx_j = expert_mapping[ej]
        
        # Load the whole [1024, seq_len] block for both experts
        sig_i = np.array(activations[idx_i], dtype=np.float32)
        sig_j = np.array(activations[idx_j], dtype=np.float32)
        
        # An expert is "activated" at token t if ANY of its 1024 neurons is non-zero
        active_i = np.any(sig_i != 0.0, axis=0)
        active_j = np.any(sig_j != 0.0, axis=0)
        
        mutual_mask = active_i & active_j
        mutual_count = np.sum(mutual_mask)
        mutual_frac = mutual_count / num_tokens
        
        status = "SUFFICIENT" if mutual_count >= THRESHOLD else "INSUFFICIENT"
        if status == "SUFFICIENT":
            total_sufficient += 1
            
        print(f"{pid:<10} {ei:<7} {ej:<7} {num_tokens:<15} {mutual_count:<15} {mutual_frac:<15.4%} {status}")

    print("\n--------------------------------------------------------------")
    print(f"Total Pairs: {len(pairs)}")
    print(f"Sufficient Support (>= {THRESHOLD} tokens): {total_sufficient}")
    print("==============================================================")

if __name__ == "__main__":
    audit_mutual_support()
