"""
run_matched_random_k.py
=======================
Phase 3: Matched Random-K Baseline Execution Harness for OLMoE-1B-7B-0924.

Scientific Research Question:
    Does CARE's capability geometry improve candidate nomination when both
    CARE and Random-K receive the exact same candidate-evaluation budget (K=5)?

Protocol:
    - Target: allenai/OLMoE-1B-7B-0924 (64 -> 48 experts across 16 steps)
    - Candidate Budget: Exactly K=5 candidate pairs evaluated per step.
    - Candidate Generation: Uniform sampling of 5 valid pairs without replacement.
    - Evaluation: Exact same 128 calibration tokens and marginal KL metric.
    - Selection: Minimum-damage candidate chosen and permanently committed.
    - Seeds: [101, 202, 303, 404, 505]
    - Checkpoints: PPL evaluated on WikiText-2 at 64, 60, 56, and 48 experts.
"""

import os
import sys
import time
import json
import random
import argparse
import hashlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.run_benchmark import load_config
from benchmark.core.evaluate import (
    prepare_wikitext_eval_batches,
    compute_ppl,
    compute_calibration_distributions,
    evaluate_marginal_kl
)
from benchmark.core.logging import BenchmarkLogger
from experiments.care_com_v21.core import PhysicalMergeEngine

def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()

def sample_valid_random_k_pairs(current_n_experts: int, k: int, rng: np.random.RandomState):
    """Samples exactly K unique valid expert pairs (i, j) with i < j without replacement."""
    all_pairs = []
    for i in range(current_n_experts):
        for j in range(i + 1, current_n_experts):
            all_pairs.append((i, j))
            
    assert len(all_pairs) >= k, f"Cannot sample {k} pairs from {len(all_pairs)} possible pairs"
    chosen_indices = rng.choice(len(all_pairs), size=k, replace=False)
    return [all_pairs[idx] for idx in chosen_indices]

def run_matched_random_k_seed(seed: int, config: dict, output_base_dir: str):
    run_dir = os.path.join(output_base_dir, f"random_k_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "trajectory.json")
    
    # Clean previous run file if resuming freshly
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                existing = json.load(f)
                if len(existing.get("trace", [])) == 16:
                    print(f"[Seed {seed}] Already completed 16 steps. Skipping.")
                    return
            except Exception:
                pass
                
    logger = BenchmarkLogger(log_path, config)
    rng = np.random.RandomState(seed)
    
    print(f"\n{'='*70}")
    print(f"STARTING MATCHED RANDOM-K RUN (Seed: {seed})")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    
    calib_path = os.path.join(
        PROJECT_ROOT, "results", "exp6c", "token_vectors", "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    df_calib = pd.read_parquet(calib_path)
    df_calib = df_calib.sample(n=128, random_state=42).reset_index(drop=True)
    
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    device = config['model']['device']
    
    print(f"[Seed {seed}] Loading model {config['model']['name']} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    
    # Record starting fingerprint
    start_hash = sha256_tensor(model.model.layers[0].mlp.experts[0].gate_proj.weight.data)
    print(f"[Seed {seed}] Model Fingerprint (Layer 0, Exp 0 Gate SHA-256): {start_hash[:16]}...")
    
    baseline_probs = compute_calibration_distributions(model, df_calib, batch_size=config['evaluation']['batch_size'])
    engine = PhysicalMergeEngine(model)
    target_experts = min(config['compression']['checkpoints'])
    
    # Base PPL
    if engine.current_num_experts in config['compression']['checkpoints']:
        base_ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
        logger.log_ppl(engine.current_num_experts, base_ppl)
        print(f"[Seed {seed}] Base PPL @ {engine.current_num_experts}: {base_ppl:.4f}")
        
    step = 1
    cumulative_kl = 0.0
    
    while engine.current_num_experts > target_experts:
        experts_before = engine.current_num_experts
        step_start_time = time.time()
        
        # 1. Nominate K=5 random valid candidate pairs
        candidate_pairs = sample_valid_random_k_pairs(experts_before, k=5, rng=rng)
        
        candidate_records = []
        min_damage = float('inf')
        best_candidate = None
        
        # 2. Physically apply and evaluate each candidate
        for i, j in candidate_pairs:
            engine.snapshot(i)
            engine.merge_experts(i, j)
            
            damage = evaluate_marginal_kl(model, df_calib, baseline_probs, batch_size=config['evaluation']['batch_size'])
            
            candidate_records.append({
                "pair": [int(i), int(j)],
                "marginal_kl": float(damage)
            })
            
            if damage < min_damage:
                min_damage = damage
                best_candidate = (i, j)
                
            engine.restore()
            
        # 3. Commit best candidate
        best_i, best_j = best_candidate
        engine.merge_experts(best_i, best_j)
        cumulative_kl += min_damage
        step_time = time.time() - step_start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        logger.log_step({
            "method": f"random_k_seed_{seed}",
            "seed": seed,
            "step": step,
            "experts_before": experts_before,
            "experts_after": experts_before - 1,
            "candidate_pairs": candidate_records,
            "selected_pair": [int(best_i), int(best_j)],
            "marginal_kl": float(min_damage),
            "cumulative_kl": float(cumulative_kl),
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"[Seed {seed}] Step {step:02d} (Experts {experts_before}->{experts_before-1}) | Best Pair: {best_i, best_j} | KL: {min_damage:.4f} | Cum KL: {cumulative_kl:.4f} | Time: {step_time:.2f}s")
        
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"[Seed {seed}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
            
        step += 1
        
    print(f"\n[Seed {seed}] Completed 16 steps. Final Expert Count: {engine.current_num_experts}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "audit_repair_pass", "pre_review_sprint", "random_k"))
    args = parser.parse_args()
    
    config = load_config()
    for s in args.seeds:
        run_matched_random_k_seed(s, config, args.output_dir)

if __name__ == "__main__":
    main()
