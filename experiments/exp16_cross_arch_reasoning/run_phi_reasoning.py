import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import copy
import gc
import json
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP15_DIR = os.path.join(PROJECT_ROOT, "exp15_merging_benchmarks")
PHI_BENCH_DIR = os.path.join(PROJECT_ROOT, "..", "benchmark", "Phi-3.5-MoE")
EXP12_DIR = os.path.join(PROJECT_ROOT, "experiment12_extracted", "experiment12", "results")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if EXP15_DIR not in sys.path:
    sys.path.insert(0, EXP15_DIR)
if PHI_BENCH_DIR not in sys.path:
    sys.path.insert(0, PHI_BENCH_DIR)

from evaluate_reasoning_core import evaluate_benchmark
try:
    from hc_smoe_engine import apply_hcsmoe_phi_layer
    from mc_smoe_engine import apply_mcsmoe_phi_layer
except ImportError:
    print("[Warning] Could not import HC-SMoE/MC-SMoE engines from exp15_merging_benchmarks")

MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"
CONFIG_PATH = os.path.join(SCRIPT_DIR, "evaluation_config.json")
RAW_GEN_DIR = os.path.join(SCRIPT_DIR, "raw_generations")
RUN_LOGS_DIR = os.path.join(SCRIPT_DIR, "run_logs")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "summary_results.csv")

os.makedirs(RAW_GEN_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    EVAL_CONFIG = json.load(f)

# -----------------------------------------------------------------------------
# 4-Bit Dequantization & Requantization Helpers for Phi-3.5-MoE
# -----------------------------------------------------------------------------

def _is_4bit(w) -> bool:
    return hasattr(w, "quant_state") and w.quant_state is not None

def _dequantize_to_float(linear: nn.Module, device) -> torch.Tensor:
    w = linear.weight
    if _is_4bit(w):
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(w.data, w.quant_state).to(torch.float32).contiguous()
    return w.data.to(torch.float32)

def _requantize_inplace(linear: nn.Module, merged_fp: torch.Tensor) -> None:
    w = linear.weight
    if _is_4bit(w):
        try:
            import bitsandbytes as bnb
            qs = w.quant_state
            blocksize = getattr(qs, "blocksize", 64)
            quant_type = getattr(qs, "quant_type", "fp4")
            new_packed, new_qs = bnb.functional.quantize_4bit(
                merged_fp.to(torch.float16),
                blocksize=blocksize,
                quant_type=quant_type,
            )
            w.data = new_packed
            w.quant_state = new_qs
            return
        except Exception:
            pass
    linear.weight = nn.Parameter(merged_fp.to(torch.bfloat16))

def merge_phi_pair_inplace(expert_a: nn.Module, expert_b: nn.Module):
    device = next(iter(expert_a.parameters())).device
    with torch.no_grad():
        for attr in ["w1", "w2", "w3"]:
            if hasattr(expert_a, attr) and hasattr(expert_b, attr):
                lin_a = getattr(expert_a, attr)
                lin_b = getattr(expert_b, attr)
                fa = _dequantize_to_float(lin_a, device)
                fb = _dequantize_to_float(lin_b, device)
                merged = (fa + fb) / 2.0
                del fa, fb
                _requantize_inplace(lin_a, merged)
                del merged

def apply_phi_merge(model, keep_idx: int, remove_idx: int):
    for layer in model.model.layers:
        moe = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
        if moe is not None and hasattr(moe, "experts"):
            merge_phi_pair_inplace(moe.experts[keep_idx], moe.experts[remove_idx])
            # Mask dead slot in router so router never routes to removed expert
            if hasattr(moe, "gate") and hasattr(moe.gate, "weight"):
                with torch.no_grad():
                    moe.gate.weight.data[remove_idx, :] = -1e9

def apply_phi_prune(model, prune_indices: List[int]):
    for layer in model.model.layers:
        moe = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
        if moe is not None and hasattr(moe, "gate") and hasattr(moe.gate, "weight"):
            with torch.no_grad():
                moe.gate.weight.data[prune_indices, :] = -1e9

# -----------------------------------------------------------------------------
# Fresh Model Loader in 4-Bit NF4
# -----------------------------------------------------------------------------

def load_fresh_phi_model() -> AutoModelForCausalLM:
    print(f"Loading fresh {MODEL_ID} in 4-bit on GPU...", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map={"": 0}
    )
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Method Applicator
# -----------------------------------------------------------------------------

def apply_phi_compression(model, method: str, target_k: int):
    if target_k == 16 or method == "Baseline":
        return

    if method == "HC-SMoE":
        for layer in model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
            apply_hcsmoe_phi_layer(moe, target_k)
        model.config.num_local_experts = target_k
        if hasattr(model.config, "num_experts"):
            model.config.num_experts = target_k
        return

    if method == "MC-SMoE":
        for layer in model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
            apply_mcsmoe_phi_layer(moe, target_k)
        model.config.num_local_experts = target_k
        if hasattr(model.config, "num_experts"):
            model.config.num_experts = target_k
        return

    # Load validated Exp 12 trajectory
    if method == "CARE-Adaptive":
        traj_file = os.path.join(EXP12_DIR, f"care_adaptive_target{target_k}_seed42.json")
        if not os.path.exists(traj_file):
            traj_file = os.path.join(PROJECT_ROOT, "benchmark", "Phi-3.5-MoE", f"care_adaptive_target{target_k}_seed42.json")
        if os.path.exists(traj_file):
            with open(traj_file) as f:
                data = json.load(f)
            traj = data.get("trajectory", [])
            for step in traj:
                pair = step["pair"]
                apply_phi_merge(model, min(pair), max(pair))
            print(f"  [CARE-Adaptive] Successfully applied {len(traj)} merged pairs from validated trajectory.", flush=True)
        else:
            print(f"  [Warning] Trajectory file not found: {traj_file}. Performing sequential adjacent merging.", flush=True)
            for k in range(16 - target_k):
                apply_phi_merge(model, 16 - 2 - k, 16 - 1 - k)

    elif method == "Sub-MoE":
        traj_file = os.path.join(EXP12_DIR, f"submoe_target{target_k}_seed42.json")
        if os.path.exists(traj_file):
            with open(traj_file) as f:
                data = json.load(f)
            traj = data.get("trajectory", [])
            for step in traj:
                pair = step["pair"]
                apply_phi_merge(model, min(pair), max(pair))
            print(f"  [Sub-MoE] Successfully applied {len(traj)} merged pairs from validated trajectory.", flush=True)
        else:
            for k in range(16 - target_k):
                apply_phi_merge(model, 16 - 2 - k, 16 - 1 - k)

    elif method == "REAP":
        reap_file = os.path.join(EXP12_DIR, f"reap_target{target_k}_seed42.json")
        if os.path.exists(reap_file):
            with open(reap_file) as f:
                data = json.load(f)
            pruned = data["trajectory"][0]["pruned_experts"]
            apply_phi_prune(model, pruned)
            print(f"  [REAP] Pruned {len(pruned)} experts: {pruned}", flush=True)
        else:
            pruned = list(range(16 - target_k))
            apply_phi_prune(model, pruned)

# -----------------------------------------------------------------------------
# Incremental Metrics Persistence
# -----------------------------------------------------------------------------

def append_summary(record: Dict[str, Any]):
    df = pd.DataFrame([record])
    if os.path.exists(SUMMARY_CSV):
        ex_df = pd.read_csv(SUMMARY_CSV)
        mask = (ex_df["model"] == record["model"]) & (ex_df["method"] == record["method"]) & \
               (ex_df["experts"] == record["experts"]) & (ex_df["benchmark"] == record["benchmark"])
        ex_df = ex_df[~mask]
        df = pd.concat([ex_df, df], ignore_index=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"  [Persisted] Summary metrics appended to {SUMMARY_CSV}", flush=True)

# -----------------------------------------------------------------------------
# Main Evaluation Loop
# -----------------------------------------------------------------------------

def main():
    print("=" * 80, flush=True)
    print("CARE-COM Exp 16 — PHI-3.5-MOE REASONING BENCHMARK RUNNER", flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    targets_plan = [
        ("Baseline", 16),
        ("CARE-Adaptive", 14),
        ("REAP", 14),
        ("Sub-MoE", 14),
        ("HC-SMoE", 14),
        ("MC-SMoE", 14),
        ("CARE-Adaptive", 12),
        ("REAP", 12),
        ("Sub-MoE", 12),
        ("HC-SMoE", 12),
        ("MC-SMoE", 12),
        ("CARE-Adaptive", 10),
        ("REAP", 10),
        ("Sub-MoE", 10),
        ("HC-SMoE", 10),
        ("MC-SMoE", 10),
        ("CARE-Adaptive", 8),
        ("REAP", 8),
        ("Sub-MoE", 8),
        ("HC-SMoE", 8),
        ("MC-SMoE", 8),
    ]

    for method, target_k in targets_plan:
        model_label = f"phi_{method.lower().replace('-', '_')}_{target_k}exp"
        gsm8k_jsonl = os.path.join(RAW_GEN_DIR, f"{model_label}_gsm8k_100q.jsonl")
        math_jsonl = os.path.join(RAW_GEN_DIR, f"{model_label}_math500_50q.jsonl")

        gsm_done = os.path.exists(gsm8k_jsonl) and sum(1 for _ in open(gsm8k_jsonl)) >= 100
        math_done = os.path.exists(math_jsonl) and sum(1 for _ in open(math_jsonl)) >= 50
        if gsm_done and math_done:
            print(f"\n>>> [Skipping] {model_label} already evaluated (100 GSM8K, 50 MATH-500 completed).", flush=True)
            continue

        print(f"\n{'#'*75}", flush=True)
        print(f"STARTING EVALUATION: Phi-3.5-MoE | Method: {method} | Retained Experts: {target_k}", flush=True)
        print(f"{'#'*75}", flush=True)

        try:
            torch.cuda.empty_cache()
            gc.collect()
            gc.collect()

            model = load_fresh_phi_model()
            apply_phi_compression(model, method, target_k)

            # Evaluate GSM8K (100 Qs)
            gsm_res = evaluate_benchmark(
                model=model,
                tokenizer=tokenizer,
                benchmark_key="gsm8k",
                model_label=model_label,
                architecture_type="instruct",
                config=EVAL_CONFIG,
                raw_gen_dir=RAW_GEN_DIR
            )
            append_summary({
                "model": "Phi-3.5-MoE",
                "method": method,
                "experts": target_k,
                **gsm_res
            })

            # Evaluate MATH-500 (50 Qs)
            math_res = evaluate_benchmark(
                model=model,
                tokenizer=tokenizer,
                benchmark_key="math500",
                model_label=model_label,
                architecture_type="instruct",
                config=EVAL_CONFIG,
                raw_gen_dir=RAW_GEN_DIR
            )
            append_summary({
                "model": "Phi-3.5-MoE",
                "method": method,
                "experts": target_k,
                **math_res
            })

            del model
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[Error] Failure during {model_label} evaluation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            del model
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()

    print("\n✓ PHI-3.5-MOE REASONING BENCHMARK COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
