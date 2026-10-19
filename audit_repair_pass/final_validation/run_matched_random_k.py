"""
audit_repair_pass/final_validation/run_matched_random_k.py
==========================================================
Phase 3 & 4: Rigorous, Audited Matched Random-K Baseline Execution Harness for OLMoE-1B-7B-0924.

Scientific Research Question:
    Does CARE's capability geometry improve candidate nomination when both
    CARE and Random-K receive the exact same candidate-evaluation budget (K=5)?

Audited Guarantees Enforced:
1. Exact parameter matching: identical checkpoint, tokenizer, calibration subset, and precision.
2. Exact evaluation matching: K=5 candidates evaluated per step with marginal KL objective.
3. Transactional reversibility: engine.snapshot() and engine.restore() ensure zero state leakage.
4. Mathematical router rule: Uses (W_r,i + W_r,j) / 2.0 to match the exact submitted code.
5. Finite numerical checks: Asserts non-NaN, non-Inf marginal KL on every candidate.
6. Bit-level fingerprinting: Computes Layer 0 SHA-256 tensor hash before and after execution.
7. Clean file logging: Overwrites trajectory with atomic JSON write; prevents aggregator concatenation.
"""

import os
import sys
import time
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmark", "OLMoE-1B-7B"))

from run_benchmark import load_config
from core.evaluate import (
    prepare_wikitext_eval_batches,
    compute_ppl,
    compute_calibration_distributions,
    evaluate_marginal_kl
)
from core.logging import BenchmarkLogger
from experiments.care_com_v21.core import PhysicalMergeEngine

def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()

def sample_valid_random_k_pairs(current_n_experts: int, k: int, rng: np.random.RandomState):
    """Samples exactly K unique valid expert pairs (i, j) with 0 <= i < j < N without replacement."""
    all_pairs = []
    for i in range(current_n_experts):
        for j in range(i + 1, current_n_experts):
            all_pairs.append((i, j))
            
    assert len(all_pairs) >= k, f"Cannot sample {k} pairs from {len(all_pairs)} possible pairs"
    chosen_indices = rng.choice(len(all_pairs), size=k, replace=False)
    selected = [all_pairs[idx] for idx in chosen_indices]
    
    # Assert validity
    for i, j in selected:
        assert 0 <= i < j < current_n_experts, f"Invalid pair ({i}, {j}) for N={current_n_experts}"
    assert len(selected) == len(set(selected)), "Duplicate candidate pairs generated"
    return selected

def run_matched_random_k_seed(seed: int, config: dict, output_base_dir: str, max_steps: int = None):
    run_dir = os.path.join(output_base_dir, f"random_k_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "trajectory.json")
    
    # Clean check if already complete
    if os.path.exists(log_path) and max_steps is None:
        try:
            with open(log_path, "r") as f:
                existing = json.load(f)
            min_check = str(min(config['compression']['checkpoints']))
            if "ppl" in existing and min_check in existing["ppl"] and len(existing.get("trace", [])) == 16:
                print(f"[Seed {seed}] Already completed all 16 steps and final PPL. Skipping.")
                return
        except Exception:
            pass
            
    # Fresh logger instance
    logger = BenchmarkLogger(log_path, config)
    rng = np.random.RandomState(seed)
    
    print(f"\n{'='*75}")
    print(f"CARE-COM FINAL VALIDATION: MATCHED RANDOM-K=5 (Seed: {seed})")
    print(f"{'='*75}")
    
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
    
    print(f"[Seed {seed}] Loading fresh model {config['model']['name']} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    
    block0 = model.model.layers[0].mlp
    if hasattr(block0.experts, 'gate_up_proj'):
        start_hash_gate = sha256_tensor(block0.experts.gate_up_proj[0].data)
    else:
        start_hash_gate = sha256_tensor(block0.experts[0].gate_proj.weight.data)
    router0 = getattr(block0, "gate", getattr(block0, "router", None))
    start_hash_router = sha256_tensor(router0.weight.data)
    print(f"[Seed {seed}] Initial Fingerprints:")
    print(f"  Layer 0 Exp 0 Gate SHA-256:   {start_hash_gate[:24]}...")
    print(f"  Layer 0 Router Weight SHA-256: {start_hash_router[:24]}...")
    
    baseline_probs = compute_calibration_distributions(model, df_calib, batch_size=config['evaluation']['batch_size'])
    engine = PhysicalMergeEngine(model)
    target_experts = min(config['compression']['checkpoints'])
    
    # Evaluate Baseline PPL at 64 experts
    if engine.current_num_experts in config['compression']['checkpoints']:
        base_ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
        logger.log_ppl(engine.current_num_experts, base_ppl)
        print(f"[Seed {seed}] Baseline PPL @ {engine.current_num_experts} experts: {base_ppl:.4f}")
        
    step = 1
    cumulative_kl = 0.0
    total_evaluations = 0
    
    while engine.current_num_experts > target_experts:
        if max_steps is not None and step > max_steps:
            print(f"[Seed {seed}] Reached max_steps={max_steps} for smoke test. Halting merge loop.")
            break
        experts_before = engine.current_num_experts
        step_start_time = time.time()
        
        # 1. Nominate exactly K=5 random valid candidate pairs
        candidate_pairs = sample_valid_random_k_pairs(experts_before, k=5, rng=rng)
        
        candidate_records = []
        min_damage = float('inf')
        best_candidate = None
        
        # 2. Physically apply each candidate merge and evaluate marginal KL
        for i, j in candidate_pairs:
            engine.snapshot(i)
            try:
                engine.merge_experts(i, j)
                damage = evaluate_marginal_kl(model, df_calib, baseline_probs, batch_size=config['evaluation']['batch_size'])
                
                # Assert numerical validity
                assert not np.isnan(damage) and not np.isinf(damage), f"Invalid KL {damage} on pair ({i}, {j})"
                total_evaluations += 1
                
                candidate_records.append({
                    "pair": [int(i), int(j)],
                    "marginal_kl": float(damage),
                    "status": "valid"
                })
                
                if damage < min_damage:
                    min_damage = damage
                    best_candidate = (i, j)
            except Exception as e:
                candidate_records.append({
                    "pair": [int(i), int(j)],
                    "marginal_kl": float('nan'),
                    "status": f"failed: {str(e)}"
                })
            finally:
                engine.restore()
                
        assert best_candidate is not None, f"All candidates failed at step {step}"
        best_i, best_j = best_candidate
        
        # 3. Permanently commit the minimum-damage candidate merge
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
            "evaluations_completed": total_evaluations,
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"[Seed {seed}] Step {step:02d} | Experts: {experts_before}->{experts_before-1} | Best Pair: [{best_i}, {best_j}] | Marginal KL: {min_damage:.4f} | Cum KL: {cumulative_kl:.4f} | Time: {step_time:.2f}s")
        
        # Checkpoint evaluation
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"  >>> PPL Checkpoint @ {engine.current_num_experts} experts: {ppl:.4f}")
            
        step += 1
        
    if hasattr(block0.experts, 'gate_up_proj'):
        final_hash = sha256_tensor(block0.experts.gate_up_proj[0].data)
    else:
        final_hash = sha256_tensor(block0.experts[0].gate_proj.weight.data)
    print(f"\n[Seed {seed}] COMPLETED SUCCESSFULLY.")
    print(f"  Final Experts: {engine.current_num_experts}")
    print(f"  Total Evaluations: {total_evaluations}")
    print(f"  Final Cumulative KL: {cumulative_kl:.4f}")
    print(f"  Final Model Fingerprint: {final_hash[:24]}...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "random_k5"))
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max merges for smoke testing")
    args = parser.parse_args()
    
    config = load_config()
    for s in args.seeds:
        run_matched_random_k_seed(s, config, args.output_dir, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
