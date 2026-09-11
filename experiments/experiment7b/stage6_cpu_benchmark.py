import os
import sys
import time
import torch
import json
import warnings
warnings.filterwarnings("ignore")

# Adjust sys path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_LAYER_IDX
from utils.model_utils import load_base_model, get_target_moe_block, apply_in_place_merge
from utils.evaluation import prepare_wikitext_eval_batches, evaluate_wikitext_oracle_kl, collect_baseline_wikitext_logits

def run_stage6():
    print("Running Stage 6: CPU Feasibility Benchmark...")
    t0 = time.time()
    model, tokenizer = load_base_model()
    load_time = time.time() - t0
    
    # 5120 tokens = 10 sequences of 512
    t0 = time.time()
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, max_tokens=5120, seq_len=512)
    baseline_logprobs = collect_baseline_wikitext_logits(model, eval_chunks)
    
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)
    
    t0 = time.time()
    restore_fn = apply_in_place_merge(moe_block, 7, 19)
    merge_time = time.time() - t0
    
    t0 = time.time()
    try:
        kl = evaluate_wikitext_oracle_kl(model, eval_chunks, baseline_logprobs)
    finally:
        restore_fn()
    inf_time = time.time() - t0
    
    total_tokens = 5120
    tok_sec = total_tokens / max(inf_time, 0.001)
    
    est_full_inf = 262144 / tok_sec
    est_full_pair = merge_time + est_full_inf
    est_18_pairs = est_full_pair * 18
    
    print(f"Benchmark complete:")
    print(f"  Model Load: {load_time:.2f} s")
    print(f"  Merge Time: {merge_time:.2f} s")
    print(f"  Inference (5120 tokens): {inf_time:.2f} s")
    print(f"  Tokens/sec: {tok_sec:.2f}")
    print(f"  Estimated Full Pair Time: {est_full_pair:.2f} s")
    print(f"  Estimated 18 Pairs Time: {est_18_pairs:.2f} s")
    
    res = {
        "load_time_s": load_time,
        "merge_time_s": merge_time,
        "inference_5120_s": inf_time,
        "tokens_per_sec": tok_sec,
        "est_full_pair_s": est_full_pair,
        "est_18_pairs_s": est_18_pairs
    }
    
    out_path = os.path.join(os.path.dirname(__file__), "../../results/exp7b/cpu_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_stage6()
