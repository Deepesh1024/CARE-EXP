"""
care_large_moe_systems/benchmark_runner.py
Master systems benchmark runner for large-expert MoE models.
Executes workload sweeps, measures timing and memory, and profiles routing statistics.
"""

import os
import sys
import json
import time
import argparse
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from workloads import WORKLOADS
from metrics_tracker import MetricsTracker
from routing_profiler import RoutingProfiler
from model_reconstruction import count_parameters, reconstruct_olmoe_model, reconstruct_qwen3_model

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
ROUTING_DIR = os.path.join(RESULTS_DIR, "routing")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "systems_benchmark_summary.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ROUTING_DIR, exist_ok=True)

def run_systems_benchmark(
    model_family: str,
    target_experts: int,
    method: str = "care_adaptive",
    device: str = "cuda",
    precision: str = "bfloat16"
):
    print("=" * 80)
    print(f"SYSTEMS BENCHMARK: {model_family.upper()} | Experts: {target_experts} | Method: {method}")
    print("=" * 80)

    tracker = MetricsTracker(device=device)
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

    # Model configuration
    if model_family == "olmoe":
        model_name = "allenai/OLMoE-1B-7B-0924"
        top_k = 8
        base_experts = 64
        traj_path = os.path.join(PROJECT_ROOT, "benchmark_results", "OLMoE-1B-7B", method, "trajectory.json")
    elif model_family == "qwen3":
        model_name = "Qwen/Qwen3-30B-A3B"
        top_k = 8
        base_experts = 128
        traj_path = os.path.join(PROJECT_ROOT, "audit", "QWEN3_CARE_COM_RESULTS.json")
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    # Step 1: Model Loading & Memory Baseline
    print(f"\n[1/5] Loading tokenizer and pristine base model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t_load_start = time.perf_counter()
    tracker.reset_peak_memory()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    tracker.synchronize()
    model_load_time_sec = round(time.perf_counter() - t_load_start, 2)
    post_load_mem = tracker.get_memory_stats()
    print(f" -> Base model loaded in {model_load_time_sec}s | Post-load VRAM: {post_load_mem['gpu_allocated_mb']:.1f} MB | CPU RAM: {post_load_mem['cpu_ram_rss_mb']:.1f} MB")

    # Step 2: Physical Replay / Compression
    replay_time_sec = 0.0
    if target_experts < base_experts:
        print(f"\n[2/5] Performing physical expert consolidation to {target_experts} experts...")
        if model_family == "olmoe":
            model, replay_time_sec = reconstruct_olmoe_model(model, target_experts, traj_path)
        elif model_family == "qwen3":
            model, replay_time_sec = reconstruct_qwen3_model(model, target_experts, traj_path)
    else:
        print(f"\n[2/5] Running pristine baseline (No compression needed).")

    param_stats = count_parameters(model, top_k=top_k)
    post_merge_mem = tracker.get_memory_stats()
    print(f" -> Parameters: Total={param_stats['total_parameters_b']}B | Expert={param_stats['expert_parameters_b']}B | Active/tok={param_stats['active_parameters_b']}B")
    print(f" -> Post-merge VRAM: {post_merge_mem['gpu_allocated_mb']:.1f} MB | Peak: {post_merge_mem['gpu_max_allocated_mb']:.1f} MB")

    # Step 3: Warmup Execution (2 queries)
    print("\n[3/5] Executing GPU warmup runs...")
    for _ in range(2):
        _ = tracker.benchmark_generation(model, tokenizer, prompt="Hello, world!", max_new_tokens=8, batch_size=1)
    tracker.synchronize()
    print(" -> Warmup completed.")

    # Step 4: Workload Latency & Throughput Sweeps (No hooks, clean timing)
    print("\n[4/5] Executing Workload Sweeps (Clean Timing & Memory)...")
    workload_results = {}

    for w_id, w_spec in WORKLOADS.items():
        print(f"\n--- Workload: {w_spec['name']} ({w_id}) ---")
        prompt = w_spec["prompt_text"]
        max_tokens = w_spec["max_new_tokens"]
        workload_results[w_id] = []

        for b in w_spec["batch_sizes"]:
            print(f"  Benchmarking batch_size={b} (in={w_spec['target_input_tokens']}, out={max_tokens})...", end="", flush=True)
            res = tracker.benchmark_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                batch_size=b
            )
            print(f" Done! TTFT: {res['ttft_ms']}ms | ITL: {res['mean_itl_ms']}ms | Decode: {res['decode_throughput_tok_sec']} tok/s | Peak VRAM: {res['peak_vram_mb']} MB")
            workload_results[w_id].append(res)

    # Step 5: Isolated Routing Profiling Pass (Hooks attached)
    print("\n[5/5] Executing Isolated Routing Profiling Pass...")
    profiler = RoutingProfiler(model, num_experts=target_experts, top_k=top_k)
    routing_stats = profiler.profile_workload(
        tokenizer=tokenizer,
        prompt=WORKLOADS["W2_medium"]["prompt_text"],
        max_new_tokens=64,
        batch_size=1
    )
    print(f" -> Routing Profiling: Mean Normalized Entropy={routing_stats['mean_normalized_routing_entropy']} | Imbalance CV={routing_stats['mean_load_imbalance_cv']}")

    # Save detailed per-model artifacts
    run_record = {
        "model_family": model_family,
        "target_experts": target_experts,
        "base_experts": base_experts,
        "method": method,
        "device": device,
        "precision": precision,
        "model_load_time_sec": model_load_time_sec,
        "replay_time_sec": round(replay_time_sec, 3),
        "parameter_stats": param_stats,
        "post_load_memory": post_load_mem,
        "post_merge_memory": post_merge_mem,
        "workload_results": workload_results,
        "routing_summary": {
            "mean_normalized_routing_entropy": routing_stats["mean_normalized_routing_entropy"],
            "mean_load_imbalance_cv": routing_stats["mean_load_imbalance_cv"]
        }
    }

    raw_filename = f"{model_family}_{method}_{target_experts}exp.json"
    with open(os.path.join(RAW_DIR, raw_filename), "w") as f:
        json.dump(run_record, f, indent=2)

    with open(os.path.join(ROUTING_DIR, f"routing_{model_family}_{method}_{target_experts}exp.json"), "w") as f:
        json.dump(routing_stats, f, indent=2)

    # Append to Summary CSV
    summary_rows = []
    for w_id, b_list in workload_results.items():
        for b_data in b_list:
            summary_rows.append({
                "model": model_family,
                "method": method,
                "experts": target_experts,
                "total_params_b": param_stats["total_parameters_b"],
                "active_params_b": param_stats["active_parameters_b"],
                "size_bf16_gb": param_stats["size_bf16_gb"],
                "workload": w_id,
                "batch_size": b_data["batch_size"],
                "prompt_tokens": b_data["prompt_tokens"],
                "generated_tokens": b_data["generated_tokens"],
                "ttft_ms": b_data["ttft_ms"],
                "mean_itl_ms": b_data["mean_itl_ms"],
                "p50_itl_ms": b_data["p50_itl_ms"],
                "p95_itl_ms": b_data["p95_itl_ms"],
                "prefill_throughput_tok_sec": b_data["prefill_throughput_tok_sec"],
                "decode_throughput_tok_sec": b_data["decode_throughput_tok_sec"],
                "e2e_throughput_tok_sec": b_data["e2e_throughput_tok_sec"],
                "peak_vram_mb": b_data["peak_vram_mb"],
                "routing_entropy": routing_stats["mean_normalized_routing_entropy"],
                "routing_imbalance_cv": routing_stats["mean_load_imbalance_cv"],
                "model_load_time_sec": model_load_time_sec,
                "replay_time_sec": round(replay_time_sec, 3)
            })

    df_new = pd.DataFrame(summary_rows)
    if os.path.exists(SUMMARY_CSV):
        df_existing = pd.read_csv(SUMMARY_CSV)
        # Avoid duplicate entries
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates(
            subset=["model", "method", "experts", "workload", "batch_size"], keep="last"
        )
        df_combined.to_csv(SUMMARY_CSV, index=False)
    else:
        df_new.to_csv(SUMMARY_CSV, index=False)

    print(f"\n[Saved] Run complete. Results saved to {raw_filename} and {SUMMARY_CSV}.")
    return run_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="olmoe", choices=["olmoe", "qwen3"])
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--method", type=str, default="care_adaptive")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bfloat16")
    args = parser.parse_args()

    run_systems_benchmark(
        model_family=args.model,
        target_experts=args.experts,
        method=args.method,
        device=args.device,
        precision=args.precision
    )
