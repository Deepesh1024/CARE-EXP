"""
audit_repair_pass/full_64e_validation/run_matched_care_64e.py
=============================================================
Audited, Versioned Full 64-Expert Matched Rematch Runner for CARE-Adaptive.

Guarantees:
1. Exact checkpoint: allenai/OLMoE-1B-7B-0924 in bfloat16.
2. Exact evaluation: K=5 candidate merges per step evaluated via marginal KL.
3. Full 64-expert nomination space: Uses care_capability_v23 to profile all active experts.
4. Identical 5 paired calibration seeds: [101, 202, 303, 404, 505].
5. Full 65,536-token WikiText-2 test PPL evaluation at checkpoints [64, 60, 56, 48].
6. Reversible transactional snapshot/restore with layer 0 tensor SHA-256 fingerprinting.
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
from audit_repair_pass.full_64e_validation.care_capability_v23 import (
    compute_global_capability_vectors_v23,
    pairwise_capability_distances_v23,
    generate_candidate_pool_v23
)

def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()

def run_matched_care_64e_seed(seed: int, config: dict, output_base_dir: str, max_steps: int = None):
    run_dir = os.path.join(output_base_dir, f"care_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "trajectory.json")
    
    # Check if already completed
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
            
    logger = BenchmarkLogger(log_path, config)
    
    print(f"\n{'='*75}")
    print(f"CARE-COM FULL 64E MATCHED REMATCH: CARE-Adaptive (Seed: {seed})")
    print(f"{'='*75}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    
    calib_path = os.path.join(
        PROJECT_ROOT, "results", "exp6c", "token_vectors", "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    df_full = pd.read_parquet(calib_path)
    df_calib = df_full.sample(n=128, random_state=seed).reset_index(drop=True)
    
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
    
    baseline_probs = compute_calibration_distributions(model, df_calib, batch_size=config['evaluation']['batch_size'], device=device)
    engine = PhysicalMergeEngine(model)
    target_experts = min(config['compression']['checkpoints'])
    
    # Evaluate Baseline PPL at 64 experts
    if engine.current_num_experts in config['compression']['checkpoints']:
        base_ppl = compute_ppl(model, eval_chunks, device=device, batch_size=config['evaluation']['batch_size'])
        logger.log_ppl(engine.current_num_experts, base_ppl)
        print(f"[Seed {seed}] Baseline PPL @ {engine.current_num_experts} experts: {base_ppl:.4f}")
        
    step = 1
    cumulative_kl = 0.0
    total_evaluations = 0
    
    while engine.current_num_experts > target_experts:
        if max_steps is not None and step > max_steps:
            print(f"[Seed {seed}] Reached max_steps={max_steps}. Halting.")
            break
            
        experts_before = engine.current_num_experts
        step_start_time = time.time()
        
        # 1. Compute Capability Vectors across all active experts (v2.3)
        t_cap_start = time.time()
        C = compute_global_capability_vectors_v23(
            model=model,
            moe_blocks=engine.moe_blocks,
            current_num_experts=experts_before,
            df_tokens=df_calib,
            samples_per_axis=20,
            device=device
        )
        t_cap = time.time() - t_cap_start
        
        # 2. Pairwise distances and top-K candidate generation across full active pool
        distances = pairwise_capability_distances_v23(C)
        candidate_pairs = generate_candidate_pool_v23(distances, pool_size=config['compression']['candidate_pool_size'])
        
        candidate_records = []
        min_damage = float('inf')
        best_candidate = None
        
        # 3. Physically apply each candidate merge and evaluate marginal KL
        for i, j in candidate_pairs:
            cap_dist = distances[i][j]
            engine.snapshot(i)
            try:
                engine.merge_experts(i, j)
                damage = evaluate_marginal_kl(model, df_calib, baseline_probs, batch_size=config['evaluation']['batch_size'], device=device)
                
                assert not np.isnan(damage) and not np.isinf(damage), f"Invalid KL {damage} on pair ({i}, {j})"
                total_evaluations += 1
                
                candidate_records.append({
                    "pair": [int(i), int(j)],
                    "capability_distance": float(cap_dist),
                    "marginal_kl": float(damage),
                    "status": "valid"
                })
                
                if damage < min_damage:
                    min_damage = damage
                    best_candidate = (i, j, cap_dist)
            except Exception as e:
                candidate_records.append({
                    "pair": [int(i), int(j)],
                    "capability_distance": float(cap_dist),
                    "marginal_kl": float('nan'),
                    "status": f"failed: {str(e)}"
                })
            finally:
                engine.restore()
                
        assert best_candidate is not None, f"All candidates failed at step {step}"
        best_i, best_j, best_cap_dist = best_candidate
        
        # 4. Permanently commit the minimum-damage candidate merge
        engine.merge_experts(best_i, best_j)
        cumulative_kl += min_damage
        step_time = time.time() - step_start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        logger.log_step({
            "method": f"care_adaptive_64e_seed_{seed}",
            "seed": seed,
            "step": step,
            "experts_before": experts_before,
            "experts_after": experts_before - 1,
            "candidate_pairs": candidate_records,
            "selected_pair": [int(best_i), int(best_j)],
            "capability_distance": float(best_cap_dist),
            "marginal_kl": float(min_damage),
            "cumulative_kl": float(cumulative_kl),
            "evaluations_completed": total_evaluations,
            "cap_time_sec": float(t_cap),
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"[Seed {seed}] Step {step:02d} | Experts: {experts_before}->{experts_before-1} | Best Pair: [{best_i}, {best_j}] | Marginal KL: {min_damage:.4f} | Cum KL: {cumulative_kl:.4f} | Cap Time: {t_cap:.1f}s | Step Time: {step_time:.1f}s")
        
        # Checkpoint evaluation
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, device=device, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"  >>> PPL Checkpoint @ {engine.current_num_experts} experts: {ppl:.4f}")
            
        step += 1
        
    print(f"\n[Seed {seed}] COMPLETED SUCCESSFULLY.")
    print(f"  Final Experts: {engine.current_num_experts}")
    print(f"  Total Evaluations: {total_evaluations}")
    print(f"  Final Cumulative KL: {cumulative_kl:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "care_adaptive_64e"))
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config()
    for s in args.seeds:
        run_matched_care_64e_seed(s, config, args.output_dir, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
