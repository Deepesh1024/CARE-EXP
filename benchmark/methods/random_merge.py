import os
import argparse
import time
import random
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
from experiments.care_com_v21.core import PhysicalMergeEngine

def load_calibration_data():
    token_path = os.path.join(
        os.path.dirname(__file__), "..", "..", 
        "results", "exp6c", "token_vectors", 
        "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    return pd.read_parquet(token_path)

def run_random_merge(seed, config):
    method_name = f"random_seed_{seed}"
    random.seed(seed)
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_results", "random", method_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = BenchmarkLogger(os.path.join(out_dir, "trajectory.json"), config)
    
    print(f"[Random Seed {seed}] Loading tokenizer and evaluation data...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    df_calibration = load_calibration_data()
    
    print(f"[Random Seed {seed}] Loading model in {config['model']['precision']}...")
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=config['model']['device']
    )
    model.eval()
    
    print(f"[Random Seed {seed}] Computing baseline calibration distributions...")
    baseline_probs = compute_calibration_distributions(model, df_calibration, batch_size=config['evaluation']['batch_size'])
    
    engine = PhysicalMergeEngine(model)
    target_experts = min(config['compression']['checkpoints'])
    
    # Checkpoint 64
    if engine.current_num_experts in config['compression']['checkpoints']:
        ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
        logger.log_ppl(engine.current_num_experts, ppl)
        print(f"[Random Seed {seed}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
        
    step = 1
    cumulative_kl = 0.0
    
    while engine.current_num_experts > target_experts:
        experts_before = engine.current_num_experts
        print(f"\n[Random Seed {seed}] Step {step} | Experts: {experts_before} -> {experts_before - 1}")
        
        step_start_time = time.time()
        
        active_indices = list(range(engine.current_num_experts))
        i, j = random.sample(active_indices, 2)
        
        # Perform permanent merge
        engine.merge_experts(i, j)
        
        # Evaluate damage purely for logging
        damage = evaluate_marginal_kl(model, df_calibration, baseline_probs, batch_size=config['evaluation']['batch_size'])
        cumulative_kl += damage
        
        step_time = time.time() - step_start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        logger.log_step({
            "method": "random",
            "seed": seed,
            "step": step,
            "experts_before": experts_before,
            "experts_after": experts_before - 1,
            "candidate_pairs": None,
            "selected_pair": [int(i), int(j)],
            "selection_score": None,
            "capability_distance": None,
            "marginal_kl": float(damage),
            "cumulative_kl": float(cumulative_kl),
            "wall_time_sec": float(step_time),
            "peak_memory_mb": float(peak_mem)
        })
        
        print(f"  -> Randomly Merged: {i, j} | KL Damage: {damage:.4f} | Time: {step_time:.2f}s")
        
        if engine.current_num_experts in config['compression']['checkpoints']:
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(engine.current_num_experts, ppl)
            print(f"[Random Seed {seed}] PPL @ {engine.current_num_experts}: {ppl:.4f}")
            
        step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    
    if args.method == "random":
        config = load_config()
        run_random_merge(args.seed, config)
