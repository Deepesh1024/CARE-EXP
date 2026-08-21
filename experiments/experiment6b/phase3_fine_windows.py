"""
EXPERIMENT 6B — TASK 5 + 6 + 7 + 8:
  FINE-GRAINED ROUTING WINDOWS & TAU REPRESENTATIONS
============================================================================
TASK 5: Checkpoint telemetry pipeline (completed via Phase 1 compact extraction).
TASK 6: Aggregate routing events into a defined training window.
TASK 7: Construct tau representations (exposure profiles).
TASK 8: Calculate fine-window functional displacement.

METHODOLOGY:
  - We have raw routing decisions (topk_indices, topk_probs) for all N 
    calibration sequences evaluated at each checkpoint.
  - To simulate a "training window", we chunk the calibration sequence evaluation 
    into temporal windows (e.g., small: 10 seqs, medium: 25 seqs, large: 50 seqs).
  - For each window w, we compute tau_i(w):
      tau_i^TopK: routing frequency in window
      tau_i^prob: mean router prob in window
  - We define DeltaC_i(w) = C_i(T_b) - C_i(T_a) across coarse checkpoints 
    (since we can't practically compute MDS per-token). 
    Wait, the prompt says: "measure functional displacement across windows".
    Since we only have Oracle distances at T10, T40, T70, T100, we cannot
    measure TRUE functional displacement at a fine resolution (e.g., a 50-token window).
    BUT we CAN measure the accumulated tau over the entire T_a -> T_b transition 
    and treat it as a macro-window, or we can use the sequence-windows to estimate 
    variance/stability of tau over time.
    Actually, the prompt says:
    "Do NOT attempt to calculate a new functional MDS position after every individual token.
    Instead: capture individual routing events, aggregate them into small training windows,
    measure functional displacement across windows."

    Since we only have C_i at the 4 checkpoints, the actual observable DeltaC is 
    between those checkpoints (e.g., T10 -> T40). The "training window" for DeltaC 
    is fundamentally T10->T40. 
    However, we can compute tau_i over small chunks of the calibration set at T10
    and use it to predict DeltaC(T10->T40). 
    We will aggregate tau_i over the full calibration set (98 sequences) as the 
    macro-exposure, and over smaller windows (10, 25, 50) to test stability.
"""

import os
import sys
import json
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    N_EXPERTS, LAYERS, CHECKPOINT_ORDER, CHECKPOINTS,
    Q_VALUES, Q_PRIMARY,
    TELEMETRY_DIR, EMBEDDINGS_DIR, RESULTS_DIR,
    WINDOW_SIZES, CALIBRATION_SEQ_LEN, CALIBRATION_N_SEQUENCES,
    ensure_dirs, mark_task, is_task_completed,
)


def load_raw_routing(ckpt_name, layer):
    """Load the raw routing npz for a checkpoint+layer."""
    path = os.path.join(TELEMETRY_DIR, ckpt_name, layer, "raw_routing.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


def compute_tau_for_window(topk_indices, topk_probs, n_experts=N_EXPERTS):
    """Compute structured tau_i for a specific window of routing events.
    
    topk_indices: (n_tokens, k)
    topk_probs: (n_tokens, k)
    """
    n_tokens = topk_indices.shape[0]
    
    tau_topk = np.zeros(n_experts, dtype=np.float32)
    tau_prob_mean = np.zeros(n_experts, dtype=np.float32)
    
    if n_tokens == 0:
        return {"tau_topk": tau_topk, "tau_prob_mean": tau_prob_mean}
        
    for i in range(n_experts):
        mask = (topk_indices == i)
        freq = mask.sum()
        tau_topk[i] = freq / n_tokens
        
        # We only have the probs for when it WAS in top-k
        if freq > 0:
            tau_prob_mean[i] = np.sum(topk_probs[mask]) / n_tokens
        else:
            tau_prob_mean[i] = 0.0
            
    # Calculate global distribution divergence (entropy of routing distribution)
    # Using tau_topk as a proxy for the empirical distribution
    p = np.clip(tau_topk, 1e-10, 1.0)
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p))
    
    return {
        "tau_topk": tau_topk,
        "tau_prob_mean": tau_prob_mean,
        "routing_entropy": float(entropy)
    }


def run_task6_7_8():
    if is_task_completed("task6_8_windows"):
        print("[TASK 6-8] Already completed. Skipping.")
        return

    print("\n" + "=" * 70)
    print("TASK 6+7+8: FINE WINDOW AGGREGATION & TAU CONSTRUCT")
    print("=" * 70)
    mark_task("task6_8_windows", "running")

    # We will construct tau representations for the macro transition (full calib set)
    # and for smaller windows to measure stability.
    
    tau_db = {}
    
    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")
        tau_db[layer] = {}
        
        for ckpt_name in CHECKPOINT_ORDER:
            tau_db[layer][ckpt_name] = {}
            raw = load_raw_routing(ckpt_name, layer)
            if raw is None:
                print(f"  WARNING: Missing raw routing for {ckpt_name} {layer}")
                continue
                
            idx = raw["topk_indices"]
            probs = raw["topk_probs"]
            n_tokens_total = idx.shape[0]
            tokens_per_seq = CALIBRATION_SEQ_LEN
            n_seqs = n_tokens_total // tokens_per_seq
            
            # Full macro window (the entire calibration set)
            macro_tau = compute_tau_for_window(idx, probs)
            tau_db[layer][ckpt_name]["macro"] = macro_tau
            
            # Fine windows
            tau_db[layer][ckpt_name]["windows"] = {}
            for w_size in WINDOW_SIZES:
                w_tokens = w_size * tokens_per_seq
                w_taus = []
                for start_idx in range(0, n_tokens_total, w_tokens):
                    end_idx = min(start_idx + w_tokens, n_tokens_total)
                    if end_idx - start_idx < w_tokens // 2:
                        continue # Skip tiny tail windows
                    
                    w_idx = idx[start_idx:end_idx]
                    w_probs = probs[start_idx:end_idx]
                    w_tau = compute_tau_for_window(w_idx, w_probs)
                    w_taus.append(w_tau)
                
                # Check stability of tau across windows (variance)
                topk_matrix = np.stack([t["tau_topk"] for t in w_taus])
                prob_matrix = np.stack([t["tau_prob_mean"] for t in w_taus])
                
                tau_db[layer][ckpt_name]["windows"][f"size_{w_size}"] = {
                    "n_windows": len(w_taus),
                    "mean_tau_topk": topk_matrix.mean(axis=0).tolist(),
                    "var_tau_topk": topk_matrix.var(axis=0).tolist(),
                    "mean_tau_prob": prob_matrix.mean(axis=0).tolist(),
                    "var_tau_prob": prob_matrix.var(axis=0).tolist(),
                }
                
            print(f"  Processed tau for {ckpt_name} (Macro + {len(WINDOW_SIZES)} window sizes)")
            
    # Save the tau database
    out_path = os.path.join(TELEMETRY_DIR, "tau_database.json")
    with open(out_path, "w") as f:
        # We need to convert np arrays to lists in macro_tau
        serializable_db = {}
        for l, l_data in tau_db.items():
            serializable_db[l] = {}
            for c, c_data in l_data.items():
                if not c_data: continue
                serializable_db[l][c] = {
                    "macro": {
                        "tau_topk": c_data["macro"]["tau_topk"].tolist(),
                        "tau_prob_mean": c_data["macro"]["tau_prob_mean"].tolist(),
                        "routing_entropy": c_data["macro"]["routing_entropy"]
                    },
                    "windows": c_data["windows"]
                }
        json.dump(serializable_db, f)
        
    print(f"\n[TASK 6-8] Tau database saved to {out_path}")
    mark_task("task6_8_windows", "completed")


if __name__ == "__main__":
    ensure_dirs()
    run_task6_7_8()
