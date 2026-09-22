import os
import argparse
import time
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.run_benchmark import load_config
from benchmark.core.evaluate import prepare_wikitext_eval_batches, compute_ppl, compute_calibration_distributions, evaluate_marginal_kl
from benchmark.core.logging import BenchmarkLogger

# Reuse validated v2.1 components
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.care_com_v21.core import PhysicalMergeEngine, generate_candidate_pool
from experiments.care_com_v21.capability import compute_global_capability_vectors, pairwise_capability_distances

def load_calibration_data():
    token_path = os.path.join(
        os.path.dirname(__file__), "..", "..", 
        "results", "exp6c", "token_vectors", 
        "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    return pd.read_parquet(token_path)

def run_care_com(method_name, config):
    is_adaptive = (method_name == "care_adaptive")
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_results", method_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = BenchmarkLogger(os.path.join(out_dir, "trajectory.json"), config)
    
    print(f"[{method_name}] Loading tokenizer and evaluation data...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    df_calibration = load_calibration_data()
    
    print(f"[{method_name}] Loading model in {config['model']['precision']}...")
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=config['model']['device']
    )
    model.eval()
    
    print(f"[{method_name}] Computing baseline calibration distributions...")
    baseline_probs = compute_calibration_distributions(model, df_calibration, batch_size=config['evaluation']['batch_size'])
    
    engine = PhysicalMergeEngine(model)
    target_experts = min(config['compression']['checkpoints'])
    
    # Checkpoint 64
    if engine.current_num_experts in config['compression']['checkpoints']:
        ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
        logger.log_ppl(engine.current_num_experts, ppl)
        print(f"[{method_name}] PPL @ {engine.current_num_experts}: {ppl:.4f}")

    # For static, compute capability only once
    if not is_adaptive:
        print(f"[{method_name}] Computing STATIC capability vectors...")
        C_static = compute_global_capability_vectors(model, df_calibration)
        
    step = 1
    cumulative_kl = 0.0
    
    while engine.current_num_experts > target_experts:
        experts_before = engine.current_num_experts
        print(f"\n[{method_name}] Step {step} | Experts: {experts_before} -> {experts_before - 1}")
        
        step_start_time = time.time()
        
        if is_adaptive:
            C_current = compute_global_capability_vectors(model, df_calibration)
        else:
            C_current = C_static
            
        distances = pairwise_capability_distances(C_current)
        candidates = generate_candidate_pool(distances, config['compression']['candidate_pool_size'])
        
        best_candidate = None
        min_damage = float('inf')
        candidate_records = []
        
        for pair, cap_dist in candidates:
            i, j = pair
            # Transactional evaluation
            engine.snapshot(i)
            engine.merge_experts(i, j)
            
            damage = evaluate_marginal_kl(model, df_calibration, baseline_probs, batch_size=config['evaluation']['batch_size'])
            
            candidate_records.append({
                "pair": [int(i), int(j)],
                "capability_distance": float(cap_dist),
                "marginal_kl": float(damage)
            })
            
            if damage < min_damage:
                min_damage = damage
                best_candidate = (i, j, cap_dist)
                
            engine.restore(i)
            
        # Perform permanent merge
        best_i, best_j, best_cap_dist = best_candidate
        engine.merge_experts(best_i, best_j)
        
        cumulative_kl += min_damage
        step_time = time.time() - step_start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        logger.log_step({
            "method": method_name,
            "seed": None,
            "step": step,
            "experts_before": experts_before,
            "experts_after": experts_before - 1,
            "candidate_pairs": [c["pair"] for c in candidate_records],
            "selected_pair": [int(best_i), int(best_j)],
            "selection_score": float(min_damage),
            "capability_distance": float(best_cap_dist),
            "marginal_kl": float(min_damage),
            "cumulative_kl": float(cumulative_kl),
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"  -> Selected Pair: {best_i, best_j} | KL Damage: {min_damage:.4f} | Time: {step_time:.2f}s")
        
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"[{method_name}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
            
        step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["care_static", "care_adaptive"])
    args = parser.parse_args()
    
    config = load_config()
    run_care_com(args.method, config)
