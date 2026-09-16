import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.experiment7b.utils.evaluation import prepare_wikitext_eval_batches
from transformers import AutoTokenizer

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

def get_global_attention_mask(num_tokens=262144):
    """Reconstructs the global attention mask used during Phase 6 extraction."""
    print("[Phase 7] Reconstructing global attention mask...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    
    # We must match the random seed used in Phase 6/7B
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, max_tokens=num_tokens)
    
    masks = []
    total = 0
    for chunk in eval_chunks:
        masks.append(chunk["attention_mask"].cpu().numpy())
        total += len(chunk["attention_mask"])
        if total >= num_tokens:
            break
            
    global_mask = np.concatenate(masks)[:num_tokens]
    return global_mask

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
    print("EXPERIMENT 7C — PHASE 7: REVISED CLOUD ANALYSIS (MUTUAL ACTIVATION)")
    print("=" * 70)
    
    # 1. Load actual merges to get the exactly 18 pairs
    merges_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7b', 'merges', 'actual_merge_results.csv')
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Missing 7B actual merge results: {merges_path}")
        
    cand_df = pd.read_csv(merges_path)
    pairs = cand_df[['pair_id', 'expert_i', 'expert_j']].to_dict('records')
    print(f"[Phase 7] Analyzing {len(pairs)} candidate pairs.")
    
    # 2. Load memmap and attention mask
    try:
        activations, meta = load_activations()
    except FileNotFoundError as e:
        print(f"[Phase 7] Blocking Error: {e}")
        return
        
    expert_mapping = meta["expert_mapping"]
    num_tokens = meta["num_tokens"]
    global_mask = get_global_attention_mask(num_tokens=num_tokens)
    
    # We only care about tokens that are NOT padding
    valid_token_mask = (global_mask == 1)
    
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
        
        # Apply padding mask
        sig_i = sig_i[:, valid_token_mask]
        sig_j = sig_j[:, valid_token_mask]
        
        # Compute mutual activation mask
        # An expert is activated if ANY of its neurons is non-zero
        active_i = np.any(sig_i != 0.0, axis=0)
        active_j = np.any(sig_j != 0.0, axis=0)
        
        mutual_mask = active_i & active_j
        mutual_count = np.sum(mutual_mask)
        mutual_frac = mutual_count / np.sum(valid_token_mask)
        
        # Restrict signatures to mutually activated tokens
        sig_i_mutual = sig_i[:, mutual_mask]
        sig_j_mutual = sig_j[:, mutual_mask]
        
        if mutual_count > 0:
            S_matrix = compute_cosine_similarity(sig_i_mutual, sig_j_mutual)
            C_i_to_j = np.mean(np.max(S_matrix, axis=1))
            C_j_to_i = np.mean(np.max(S_matrix, axis=0))
            C_mutual = 0.5 * (C_i_to_j + C_j_to_i)
        else:
            C_i_to_j = 0.0
            C_j_to_i = 0.0
            C_mutual = 0.0
        
        results.append({
            'pair_id': pair_id,
            'expert_i': ei,
            'expert_j': ej,
            'mutual_count': mutual_count,
            'mutual_frac': mutual_frac,
            'C_i_to_j': C_i_to_j,
            'C_j_to_i': C_j_to_i,
            'C_mutual': C_mutual
        })
        
    out_dir = os.path.join(RESULTS_DIR_7C, "revised", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cloud_analysis_results.csv")
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    
    print("\n[Phase 7] Revised Cloud Analysis Complete!")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_cloud_analysis()
