"""
benchmark/OLMoE-1B-7B/methods/random_merge.py
============================================================
Audited Random Merging Baseline for OLMoE-1B-7B.

Supports:
1. Random-K5 (Faithful, Matched Protocol):
   At each step, samples K=5 random pairs of active experts, evaluates each
   candidate physically via marginal KL on the calibration subset, and permanently
   commits the candidate with minimal functional damage. This precisely matches
   CARE-COM's candidate evaluation budget and objective function.
2. Random-1 (Historical, Unbudgeted Baseline):
   Selects 1 random pair and commits without candidate evaluation (retained for
   historical ablation verification).
"""

import os
import sys
import argparse
import time
import random
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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

def load_calibration_data():
    token_path = os.path.join(
        PROJECT_ROOT, 
        "results", "exp6c", "token_vectors", 
        "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    return pd.read_parquet(token_path)

def run_random_merge(seed: int, config: dict, candidate_k: int = 5):
    method_name = f"random_k{candidate_k}_seed_{seed}" if candidate_k > 1 else f"random_seed_{seed}"
    random.seed(seed)
    torch.manual_seed(seed)
    
    out_dir = os.path.join(PROJECT_ROOT, "benchmark_results", "random", method_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = BenchmarkLogger(os.path.join(out_dir, "trajectory.json"), config)
    
    print(f"[{method_name}] Loading tokenizer and evaluation data...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    df_calibration = load_calibration_data()
    # Sample 128 sequences using matched seed
    df_calibration = df_calibration.sample(n=128, random_state=seed).reset_index(drop=True)
    
    print(f"[{method_name}] Loading model in {config['model']['precision']}...")
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    device = config['model']['device']
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    
    print(f"[{method_name}] Computing baseline calibration distributions...")
    baseline_probs = compute_calibration_distributions(
        model, df_calibration, batch_size=config['evaluation']['batch_size'], device=device
    )
    
    engine = PhysicalMergeEngine(model)
    target_experts = min(config['compression']['checkpoints'])
    
    # Checkpoint 64
    if engine.current_num_experts in config['compression']['checkpoints']:
        ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'], device=device)
        logger.log_ppl(engine.current_num_experts, ppl)
        print(f"[{method_name}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
        
    step = 1
    cumulative_kl = 0.0
    
    while engine.current_num_experts > target_experts:
        experts_before = engine.current_num_experts
        print(f"\n[{method_name}] Step {step} | Experts: {experts_before} -> {experts_before - 1}")
        step_start_time = time.time()
        
        # Formulate candidate pool of active expert pairs
        all_possible_pairs = [
            (i, j) for i in range(experts_before) for j in range(i + 1, experts_before)
        ]
        
        num_candidates = min(candidate_k, len(all_possible_pairs))
        candidate_pool = random.sample(all_possible_pairs, num_candidates)
        
        candidate_records = []
        best_candidate = None
        min_damage = float('inf')
        
        if candidate_k > 1:
            # Transactionally evaluate K candidates
            for i, j in candidate_pool:
                engine.snapshot(i, j)
                try:
                    engine.merge_experts(i, j)
                    damage = evaluate_marginal_kl(
                        model, df_calibration, baseline_probs,
                        batch_size=config['evaluation']['batch_size'], device=device
                    )
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
        else:
            # Random-1: select single random pair without evaluation
            best_candidate = candidate_pool[0]
            engine.snapshot(best_candidate[0], best_candidate[1])
            engine.merge_experts(best_candidate[0], best_candidate[1])
            min_damage = evaluate_marginal_kl(
                model, df_calibration, baseline_probs,
                batch_size=config['evaluation']['batch_size'], device=device
            )
            engine.restore()
            candidate_records.append({
                "pair": [int(best_candidate[0]), int(best_candidate[1])],
                "marginal_kl": float(min_damage),
                "status": "valid"
            })
            
        selected_i, selected_j = best_candidate
        
        # Permanently commit the selected merge
        engine.merge_experts(selected_i, selected_j)
        cumulative_kl += min_damage
        step_time = time.time() - step_start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        logger.log_step({
            "method": method_name,
            "seed": seed,
            "step": step,
            "experts_before": experts_before,
            "experts_after": experts_before - 1,
            "candidate_pairs": candidate_records,
            "selected_pair": [int(selected_i), int(selected_j)],
            "marginal_kl": float(min_damage),
            "cumulative_kl": float(cumulative_kl),
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"  -> Merged: ({selected_i}, {selected_j}) | KL Damage: {min_damage:.4f} | Time: {step_time:.2f}s")
        
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'], device=device)
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"[{method_name}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
            
        step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate_k", type=int, default=5, help="Candidate budget: 5 for Random-K5, 1 for Random-1")
    args = parser.parse_args()
    
    config = load_config()
    run_random_merge(args.seed, config, candidate_k=args.candidate_k)
