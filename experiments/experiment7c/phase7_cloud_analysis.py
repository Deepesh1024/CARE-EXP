import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

RESULTS_DIR_7C = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7c'))

def load_activations():
    out_dir = os.path.join(RESULTS_DIR_7C, "activations")
    memmap_path = os.path.join(out_dir, "expert_signatures.npy")
    memmap_meta_path = os.path.join(out_dir, "expert_signatures_meta.json")
    
    if not os.path.exists(memmap_meta_path):
        raise FileNotFoundError(f"Activations not found. Run phase 6 first. Missing: {memmap_meta_path}")
        
    with open(memmap_meta_path, "r") as f:
        meta = json.load(f)
        
    shape = tuple(meta["shape"])
    dtype = np.float16 if meta["dtype"] == "float16" else np.float32
    
    activations = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
    return activations, meta

def compute_cosine_similarity(sig_i, sig_j):
    # L2 normalize first (Normalization Protocol B)
    # Adding small epsilon to avoid div by zero
    norm_i = np.linalg.norm(sig_i, axis=1, keepdims=True) + 1e-10
    norm_j = np.linalg.norm(sig_j, axis=1, keepdims=True) + 1e-10
    
    sig_i_norm = sig_i / norm_i
    sig_j_norm = sig_j / norm_j
    
    # Cosine similarity matrix: [1024, 1024]
    # sig_i_norm: [1024, seq_len]
    # sig_j_norm: [1024, seq_len]
    return np.dot(sig_i_norm, sig_j_norm.T)

def run_cloud_analysis():
    print("=" * 70)
    print("EXPERIMENT 7C — PHASE 7: CLOUD ANALYSIS")
    print("=" * 70)
    
    # 1. Load actual merges to get the exactly 18 pairs
    merges_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7b', 'merges', 'actual_merge_results.csv')
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Missing 7B actual merge results: {merges_path}")
        
    cand_df = pd.read_csv(merges_path)
    pairs = cand_df[['pair_id', 'expert_i', 'expert_j']].to_dict('records')
    print(f"[Phase 7] Analyzing {len(pairs)} candidate pairs.")
    
    # 2. Load memmap
    try:
        activations, meta = load_activations()
    except FileNotFoundError as e:
        print(f"[Phase 7] Blocking Error: {e}")
        return
        
    expert_mapping = meta["expert_mapping"]
    
    results = []
    
    for pair in tqdm(pairs, desc="Computing similarities"):
        pair_id = pair['pair_id']
        ei, ej = pair['expert_i'], pair['expert_j']
        
        idx_i = expert_mapping[str(ei)]
        idx_j = expert_mapping[str(ej)]
        
        # Extract [1024, num_tokens]
        # We copy to memory as float32 for stable math
        sig_i = np.array(activations[idx_i], dtype=np.float32)
        sig_j = np.array(activations[idx_j], dtype=np.float32)
        
        S_matrix = compute_cosine_similarity(sig_i, sig_j)
        
        # Mutual coverage definitions
        C_i_to_j = np.mean(np.max(S_matrix, axis=1)) # max over j, mean over i
        C_j_to_i = np.mean(np.max(S_matrix, axis=0)) # max over i, mean over j
        C_mutual = 0.5 * (C_i_to_j + C_j_to_i)
        
        results.append({
            'pair_id': pair_id,
            'expert_i': ei,
            'expert_j': ej,
            'C_i_to_j': C_i_to_j,
            'C_j_to_i': C_j_to_i,
            'C_mutual': C_mutual
        })
        
    out_dir = os.path.join(RESULTS_DIR_7C, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cloud_analysis_results.csv")
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    
    print("\n[Phase 7] Cloud Analysis Complete!")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_cloud_analysis()
