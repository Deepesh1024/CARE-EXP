"""
benchmark/OLMoE-1B-7B/methods/care_com.py
============================================================
Audited and Corrected CARE-COM Benchmark Runner for OLMoE-1B-7B.

Supports:
1. CARE-Adaptive:
   Dynamically recomputes capability state C_t across all active experts,
   nominates top-K nearest active pairs, evaluates marginal KL damage physically,
   and commits argmin damage.
2. CARE-Static:
   Computes capability geometry C_0 once at N=64, freezes the pair ranking,
   and sequentially commits the highest ranked valid pair, mapping original
   expert indices to active module indices as removals occur.
"""

import os
import sys
import argparse
import time
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

# Use audited v2.1/v2.3 components
from experiments.care_com_v21.core import PhysicalMergeEngine, generate_candidate_pool
from experiments.care_com_v21.capability import compute_global_capability_vectors, pairwise_capability_distances

def load_calibration_data():
    token_path = os.path.join(
        PROJECT_ROOT, 
        "results", "exp6c", "token_vectors", 
        "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    return pd.read_parquet(token_path)

def get_frozen_ranking(C):
    """Sorts all pairs based on capability distance from the initial state."""
    distances = pairwise_capability_distances(C)
    pairs = []
    seen = set()
    num_experts = C.shape[0]
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            if j in distances[i]:
                pairs.append((i, j, distances[i][j]))
    pairs.sort(key=lambda x: x[2])
    return pairs

def map_original_indices_to_current(orig_i, orig_j, removed_experts):
    """Maps original expert indices to active module list indices."""
    if orig_i in removed_experts or orig_j in removed_experts:
        return None
    cur_i = orig_i - sum(1 for r in removed_experts if r < orig_i)
    cur_j = orig_j - sum(1 for r in removed_experts if r < orig_j)
    return min(cur_i, cur_j), max(cur_i, cur_j)

def run_care_com(method_name: str, config: dict, seed: int = 42):
    is_adaptive = (method_name == "care_adaptive")
    out_dir = os.path.join(PROJECT_ROOT, "benchmark_results", method_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = BenchmarkLogger(os.path.join(out_dir, "trajectory.json"), config)
    
    print(f"[{method_name}] Loading tokenizer and evaluation data...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    df_calibration = load_calibration_data()
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

    # For static, compute capability only once and freeze ranking
    frozen_ranking = None
    removed_experts = set()
    ranking_idx = 0
    if not is_adaptive:
        print(f"[{method_name}] Computing STATIC capability vectors at N=64...")
        C_static = compute_global_capability_vectors(
            model, engine.moe_blocks, df_calibration, device=device, current_num_experts=engine.current_num_experts
        )
        frozen_ranking = get_frozen_ranking(C_static)
        
    step = 1
    cumulative_kl = 0.0
    
    while engine.current_num_experts > target_experts:
        experts_before = engine.current_num_experts
        print(f"\n[{method_name}] Step {step} | Experts: {experts_before} -> {experts_before - 1}")
        step_start_time = time.time()
        
        if is_adaptive:
            # CARE-Adaptive: recompute capability and evaluate candidate pool
            C_current = compute_global_capability_vectors(
                model, engine.moe_blocks, df_calibration, device=device, current_num_experts=experts_before
            )
            distances = pairwise_capability_distances(C_current)
            candidates = generate_candidate_pool(distances, config['compression']['candidate_pool_size'])
            
            best_candidate = None
            min_damage = float('inf')
            candidate_records = []
            
            for i, j in candidates:
                cap_dist = distances[i][j]
                engine.snapshot(i, j)
                damage = evaluate_marginal_kl(
                    model, df_calibration, baseline_probs,
                    batch_size=config['evaluation']['batch_size'], device=device
                )
                candidate_records.append({
                    "pair": [int(i), int(j)],
                    "capability_distance": float(cap_dist),
                    "marginal_kl": float(damage)
                })
                if damage < min_damage:
                    min_damage = damage
                    best_candidate = (i, j, cap_dist)
                engine.restore()
                
            best_i, best_j, best_cap_dist = best_candidate
        else:
            # CARE-Static: find next valid pair from frozen ranking
            valid_pair = None
            while ranking_idx < len(frozen_ranking):
                orig_i, orig_j, cap_dist = frozen_ranking[ranking_idx]
                mapped = map_original_indices_to_current(orig_i, orig_j, removed_experts)
                if mapped is not None:
                    valid_pair = {
                        "original": (orig_i, orig_j),
                        "current": mapped,
                        "capability_distance": cap_dist,
                        "rank": ranking_idx
                    }
                    ranking_idx += 1
                    break
                ranking_idx += 1
                
            if valid_pair is None:
                raise RuntimeError("Exhausted frozen ranking before reaching target expert count!")
                
            best_i, best_j = valid_pair["current"]
            orig_i, orig_j = valid_pair["original"]
            best_cap_dist = valid_pair["capability_distance"]
            
            # Evaluate damage purely for logging comparison
            engine.snapshot(best_i, best_j)
            engine.merge_experts(best_i, best_j)
            min_damage = evaluate_marginal_kl(
                model, df_calibration, baseline_probs,
                batch_size=config['evaluation']['batch_size'], device=device
            )
            engine.restore()
            candidate_records = [{
                "pair": [int(best_i), int(best_j)],
                "original_pair": [int(orig_i), int(orig_j)],
                "capability_distance": float(best_cap_dist),
                "marginal_kl": float(min_damage)
            }]
            removed_experts.add(orig_j)
            
        # Perform permanent merge
        engine.merge_experts(best_i, best_j)
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
            "selected_pair": [int(best_i), int(best_j)],
            "capability_distance": float(best_cap_dist),
            "marginal_kl": float(min_damage),
            "cumulative_kl": float(cumulative_kl),
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"  -> Merged: ({best_i}, {best_j}) | KL Damage: {min_damage:.4f} | Time: {step_time:.2f}s")
        
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'], device=device)
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"[{method_name}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
            
        step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["care_static", "care_adaptive"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = load_config()
    run_care_com(args.method, config, seed=args.seed)
