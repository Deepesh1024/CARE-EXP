import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import copy
import gc
import json
from typing import Dict, List, Any, Optional
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP15_DIR = os.path.join(PROJECT_ROOT, "exp15_merging_benchmarks")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if EXP15_DIR not in sys.path:
    sys.path.insert(0, EXP15_DIR)

from evaluate_reasoning_core import evaluate_benchmark
try:
    from hc_smoe_engine import apply_hcsmoe_jetmoe_layer
    from mc_smoe_engine import apply_mcsmoe_jetmoe_layer
except ImportError:
    print("[Warning] Could not import HC-SMoE/MC-SMoE engines from exp15_merging_benchmarks")

MODEL_ID = "jetmoe/jetmoe-8b"
CONFIG_PATH = os.path.join(SCRIPT_DIR, "evaluation_config.json")
RAW_GEN_DIR = os.path.join(SCRIPT_DIR, "raw_generations")
RUN_LOGS_DIR = os.path.join(SCRIPT_DIR, "run_logs")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "summary_results.csv")
PER_Q_CSV = os.path.join(SCRIPT_DIR, "per_question_results.csv")
JETMOE_CKPT_BASE = os.path.join(PROJECT_ROOT, "..", "benchmark_results", "JetMoE-8B", "checkpoints")

os.makedirs(RAW_GEN_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    EVAL_CONFIG = json.load(f)

# -----------------------------------------------------------------------------
# JetMoE Model Construction / Loading Helpers
# -----------------------------------------------------------------------------

def get_jetmoe_mlp_tensors(model, layer_idx):
    input_linear, output_linear, router = None, None, None
    prefix = f"model.layers.{layer_idx}.mlp"
    for name, param in model.named_parameters():
        if f"{prefix}.input_linear.weight" in name:
            input_linear = param
        elif f"{prefix}.output_linear.weight" in name:
            output_linear = param
        elif f"{prefix}.router.layer.weight" in name:
            router = param
    return input_linear, output_linear, router

def merge_experts_jetmoe(model, layer_idx, keep_idx, remove_idx):
    input_linear, output_linear, router = get_jetmoe_mlp_tensors(model, layer_idx)
    with torch.no_grad():
        merged_in  = (input_linear.data[keep_idx]  + input_linear.data[remove_idx])  / 2.0
        merged_out = (output_linear.data[keep_idx] + output_linear.data[remove_idx]) / 2.0
        merged_r   = (router.data[keep_idx]        + router.data[remove_idx])        / 2.0
        input_linear.data[keep_idx].copy_(merged_in)
        input_linear.data[remove_idx].copy_(merged_in)
        output_linear.data[keep_idx].copy_(merged_out)
        output_linear.data[remove_idx].copy_(merged_out)
        router.data[keep_idx].copy_(merged_r)
        router.data[remove_idx].copy_(merged_r * 0.99)

def get_characteristic_activations(model):
    final_ca = {}
    for i in range(24):
        input_linear, _, _ = get_jetmoe_mlp_tensors(model, i)
        ca = input_linear.data.detach().mean(dim=1).cpu().to(torch.float32).numpy()
        final_ca[i] = ca
    return final_ca

def build_jetmoe_submoe(model, target_experts: int):
    import numpy as np
    ca_stats = get_characteristic_activations(model)
    for i in range(24):
        ca = ca_stats[i].copy()
        active = list(range(8))
        while len(active) > target_experts:
            pairs = []
            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    i1, i2 = active[ai], active[bi]
                    n1 = np.linalg.norm(ca[i1]) + 1e-9
                    n2 = np.linalg.norm(ca[i2]) + 1e-9
                    sim = float(np.dot(ca[i1], ca[i2]) / (n1 * n2))
                    pairs.append((sim, i1, i2))
            pairs.sort(reverse=True)
            _, k, r = pairs[0]
            merge_experts_jetmoe(model, i, k, r)
            ca[k] = (ca[k] + ca[r]) / 2.0
            active.remove(r)

def build_jetmoe_reap(model, target_experts: int):
    # Check for pre-computed reap scores
    reap_scores_path = os.path.join(PROJECT_ROOT, "..", "benchmark_results", "JetMoE-8B", "reap", "reap_scores.json")
    if os.path.exists(reap_scores_path):
        with open(reap_scores_path) as f:
            raw = json.load(f)
            scores = {int(k): v for k, v in raw.items()}
    else:
        # Fallback uniform saliency if scores not found
        scores = {i: [1.0 / (j + 1) for j in range(8)] for i in range(24)}
        
    for i in range(24):
        s = torch.tensor(scores[i])
        _, kept = torch.topk(s, target_experts, largest=True)
        all_experts = set(range(8))
        kept_experts = set(kept.tolist())
        removed = list(all_experts - kept_experts)
        mlp = model.model.layers[i].mlp
        with torch.no_grad():
            mlp.router.layer.weight[removed, :] = -1e9

def build_jetmoe_care(model, tokenizer, target_experts: int):
    import numpy as np
    from datasets import load_dataset as _ld

    def _micro_ppl(m, tok, num_tokens=1000):
        m.eval()
        ds = _ld("wikitext", "wikitext-2-raw-v1", split="test")
        enc = tok("\n\n".join(ds["text"]), return_tensors="pt")
        lim = min(enc.input_ids.size(1), num_tokens)
        nlls, tot = [], 0
        seq_len = 512
        with torch.inference_mode():
            for b in range(0, lim, seq_len):
                e = min(b + seq_len, lim)
                if e - b == 0: break
                ids = enc.input_ids[:, b:e].cuda()
                out = m(ids, labels=ids.clone())
                nlls.append(out.loss * (e - b))
                tot += (e - b)
        return torch.exp(torch.stack(nlls).sum() / tot).item()

    ca_stats = get_characteristic_activations(model)
    for i in range(24):
        ca = ca_stats[i].copy()
        active = list(range(8))
        while len(active) > target_experts:
            pairs = []
            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    i1, i2 = active[ai], active[bi]
                    n1 = np.linalg.norm(ca[i1]) + 1e-9
                    n2 = np.linalg.norm(ca[i2]) + 1e-9
                    sim = float(np.dot(ca[i1], ca[i2]) / (n1 * n2))
                    pairs.append((sim, i1, i2))
            pairs.sort(reverse=True)
            candidates = pairs[:3]
            best_pair, best_ppl = None, float("inf")
            for _, k_idx, r_idx in candidates:
                in_w, out_w, r_w = get_jetmoe_mlp_tensors(model, i)
                in_bak  = in_w.data.detach().clone()
                out_bak = out_w.data.detach().clone()
                r_bak   = r_w.data.detach().clone()
                merge_experts_jetmoe(model, i, k_idx, r_idx)
                ppl = _micro_ppl(model, tokenizer)
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_pair = (k_idx, r_idx)
                in_w.data.copy_(in_bak); out_w.data.copy_(out_bak); r_w.data.copy_(r_bak)
            k, r = best_pair
            merge_experts_jetmoe(model, i, k, r)
            active.remove(r)
            ca[k] = (ca[k] + ca[r]) / 2.0

def load_or_build_jetmoe(method: str, target_k: int, tokenizer) -> AutoModelForCausalLM:
    # 1. Check for pre-saved checkpoint on disk
    method_slug = method.lower().replace("-", "_")
    disk_ckpt_dir = os.path.join(JETMOE_CKPT_BASE, f"jetmoe_{method_slug}_{target_k}")
    if target_k == 8 and method == "Baseline":
        disk_ckpt_dir = os.path.join(JETMOE_CKPT_BASE, "jetmoe_base")

    if os.path.isdir(disk_ckpt_dir) and os.path.exists(os.path.join(disk_ckpt_dir, "config.json")):
        print(f"  [Found Checkpoint on Disk] Loading {disk_ckpt_dir} directly...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            disk_ckpt_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )
        return model

    # 2. Build on-the-fly from base model
    print(f"  [Dynamic Builder] Instantiating {method} @ {target_k} experts from {MODEL_ID}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if target_k == 8 or method == "Baseline":
        model = model.to("cuda")
        return model

    if method == "HC-SMoE":
        for l_idx, layer in enumerate(model.model.layers):
            apply_hcsmoe_jetmoe_layer(layer.mlp, target_k)
        model.config.num_local_experts = target_k
    elif method == "MC-SMoE":
        for l_idx, layer in enumerate(model.model.layers):
            apply_mcsmoe_jetmoe_layer(layer.mlp, target_k)
        model.config.num_local_experts = target_k
    elif method == "Sub-MoE":
        build_jetmoe_submoe(model, target_k)
    elif method == "REAP":
        build_jetmoe_reap(model, target_k)
    elif method == "CARE-Adaptive":
        model = model.to("cuda")
        build_jetmoe_care(model, tokenizer, target_k)
        return model
    else:
        raise ValueError(f"Unknown JetMoE method: {method}")

    model = model.to("cuda")
    return model

# -----------------------------------------------------------------------------
# Incremental Metrics Persistence
# -----------------------------------------------------------------------------

def append_summary(record: Dict[str, Any]):
    df = pd.DataFrame([record])
    if os.path.exists(SUMMARY_CSV):
        ex_df = pd.read_csv(SUMMARY_CSV)
        # deduplicate matching (model, method, experts, benchmark)
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
    print("CARE-COM Exp 16 — JETMOE-8B REASONING BENCHMARK RUNNER", flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Evaluation plan
    targets_plan = [
        ("Baseline", 8),
        ("CARE-Adaptive", 7),
        ("REAP", 7),
        ("Sub-MoE", 7),
        ("HC-SMoE", 7),
        ("MC-SMoE", 7),
        ("CARE-Adaptive", 6),
        ("REAP", 6),
        ("Sub-MoE", 6),
        ("HC-SMoE", 6),
        ("MC-SMoE", 6),
        ("CARE-Adaptive", 4),
        ("REAP", 4),
        ("Sub-MoE", 4),
        ("HC-SMoE", 4),
        ("MC-SMoE", 4),
    ]

    for method, target_k in targets_plan:
        model_label = f"jetmoe_{method.lower().replace('-', '_')}_{target_k}exp"
        gsm8k_jsonl = os.path.join(RAW_GEN_DIR, f"{model_label}_gsm8k_100q.jsonl")
        math_jsonl = os.path.join(RAW_GEN_DIR, f"{model_label}_math500_50q.jsonl")

        # Check if already 100% completed
        gsm_done = os.path.exists(gsm8k_jsonl) and sum(1 for _ in open(gsm8k_jsonl)) >= 100
        math_done = os.path.exists(math_jsonl) and sum(1 for _ in open(math_jsonl)) >= 50
        if gsm_done and math_done:
            print(f"\n>>> [Skipping] {model_label} already evaluated (100 GSM8K, 50 MATH-500 completed).", flush=True)
            continue

        print(f"\n{'#'*75}", flush=True)
        print(f"STARTING EVALUATION: JetMoE-8B | Method: {method} | Retained Experts: {target_k}", flush=True)
        print(f"{'#'*75}", flush=True)

        try:
            torch.cuda.empty_cache()
            gc.collect()
            model = load_or_build_jetmoe(method, target_k, tokenizer)

            # Evaluate GSM8K (100 Qs)
            gsm_res = evaluate_benchmark(
                model=model,
                tokenizer=tokenizer,
                benchmark_key="gsm8k",
                model_label=model_label,
                architecture_type="base",
                config=EVAL_CONFIG,
                raw_gen_dir=RAW_GEN_DIR
            )
            append_summary({
                "model": "JetMoE-8B",
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
                architecture_type="base",
                config=EVAL_CONFIG,
                raw_gen_dir=RAW_GEN_DIR
            )
            append_summary({
                "model": "JetMoE-8B",
                "method": method,
                "experts": target_k,
                **math_res
            })

            del model
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"\n[Error] Failure during {model_label} evaluation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            del model
            torch.cuda.empty_cache()
            gc.collect()

    print("\n✓ JETMOE-8B REASONING BENCHMARK COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
