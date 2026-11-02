"""
benchmark/run_l40s_benchmark.py
================================
Unified, rigorous benchmark orchestrator for CARE-MoE and baseline methods on NVIDIA L40S.

Features:
- Invokes all compression algorithms strictly through the benchmark.compressors OOP interface.
- Measures reference baseline (64 experts): PPL, throughput (tokens/s), latency (ms/batch), VRAM, params.
- Compresses to targets [60, 56, 48] experts.
- Verifies expert counts across all 16 layers.
- Measures post-compression PPL, functional KL divergence, parameter reduction, and inference speed.
- Executes strict checkpoint save and reload parity checks (missing=0, unexpected=0, mismatched=0, max_diff < 1e-2).
- Emits unique timestamped run artifacts, JSON manifests, and Markdown reports.
"""

from __future__ import annotations
import os
import sys
import time
import json
import uuid
import shutil
import argparse
import datetime
import gc
import copy
import hashlib
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure benchmark/OLMoE-1B-7B and methods are importable
_OLMOE_BENCH_DIR = PROJECT_ROOT / "benchmark" / "OLMoE-1B-7B"
if str(_OLMOE_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_OLMOE_BENCH_DIR))

_FAITHFUL_DIR = _OLMOE_BENCH_DIR / "methods"
if str(_FAITHFUL_DIR) not in sys.path:
    sys.path.insert(0, str(_FAITHFUL_DIR))

from benchmark.compressors import get_compressor, BaseMoEModelAdapter, OLMoEModelAdapter, CompressionResult
from core.evaluate import (
    prepare_wikitext_eval_batches,
    compute_ppl,
    compute_calibration_distributions,
    evaluate_marginal_kl,
    compute_heldout_kl,
    compute_causal_kl_from_logits,
)

CRITICAL_SOURCE_FILES = [
    "benchmark/run_l40s_benchmark.py",
    "experiments/care_com_v21/core.py",
    "experiments/care_com_v22/adaptive.py",
    "experiments/care_com_v22/baselines.py",
    "benchmark/compressors/base.py",
    "benchmark/compressors/care_static.py",
    "benchmark/compressors/care_adaptive.py",
    "benchmark/compressors/random_control.py",
    "benchmark/compressors/sub_moe.py",
    "benchmark/compressors/reap_prune.py",
    "benchmark/OLMoE-1B-7B/core/evaluate.py",
]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_git_info() -> Dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        dirty_lines = [l for l in status_res.stdout.strip().split("\n") if l.strip()]
        return {
            "revision": rev,
            "is_dirty": len(dirty_lines) > 0,
            "dirty_files": dirty_lines,
        }
    except Exception as e:
        return {
            "revision": "unknown",
            "is_dirty": True,
            "dirty_files": [],
            "error": str(e),
        }


def get_critical_source_hashes() -> Dict[str, str]:
    hashes = {}
    for rel_path in CRITICAL_SOURCE_FILES:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            hashes[rel_path] = h.hexdigest()
        else:
            hashes[rel_path] = "FILE_NOT_FOUND"
    return hashes


def create_unique_run_dir(base_output_dir: Path) -> Path:
    """
    Creates a collision-resistant run directory using UTC timestamp and UUID.
    Guarantees never overwriting an existing directory (exist_ok=False).
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        unique_token = uuid.uuid4().hex[:8]
        candidate = base_output_dir / f"run_{ts}_{unique_token}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            time.sleep(0.01)
            continue
    raise RuntimeError(f"Failed to create collision-resistant run directory under {base_output_dir}")


def load_and_validate_calibration_data(
    parquet_path: Path,
    required_samples: int,
    required_columns: Tuple[str, ...] = ("input_ids", "attention_mask", "axis_idx"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates calibration parquet artifact before loading any model weights.
    Aborts immediately if missing, malformed, empty, or missing required fields.
    Synthetic fallback is strictly removed from the benchmark.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"[FATAL CALIBRATION ERROR] Required calibration artifact does not exist: {parquet_path}. "
            f"Synthetic fallback has been permanently removed from the research benchmark."
        )

    h = hashlib.sha256()
    with open(parquet_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    cal_file_sha256 = h.hexdigest()

    try:
        df_cal = pd.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(
            f"[FATAL CALIBRATION ERROR] Calibration parquet {parquet_path} is malformed or unreadable: {e}"
        ) from e

    if len(df_cal) == 0:
        raise ValueError(
            f"[FATAL CALIBRATION ERROR] Calibration parquet {parquet_path} is empty (0 rows)."
        )

    missing_cols = [c for c in required_columns if c not in df_cal.columns]
    if missing_cols:
        raise ValueError(
            f"[FATAL CALIBRATION ERROR] Calibration parquet {parquet_path} lacks required columns: {missing_cols}. "
            f"Available columns: {list(df_cal.columns)}"
        )

    if len(df_cal) < required_samples:
        raise ValueError(
            f"[FATAL CALIBRATION ERROR] Calibration parquet has only {len(df_cal)} rows, but "
            f"--cal-samples requested {required_samples} samples."
        )

    cal_meta = {
        "path": str(parquet_path.resolve()),
        "sha256": cal_file_sha256,
        "total_rows": len(df_cal),
        "required_columns": list(required_columns),
        "cal_samples_requested": required_samples,
    }
    return df_cal, cal_meta


def measure_inference_speed(
    model: nn.Module,
    eval_chunks: List[Dict[str, torch.Tensor]],
    device: str = "cuda:0",
    batch_size: int = 4,
    timed_batches: int = 16,
) -> Tuple[float, float]:
    """
    Measures inference throughput (tokens/sec) and latency (ms/batch).
    Uses warm-up passes followed by CUDA synchronization for accurate timing.
    """
    model.eval()
    batches = []
    for i in range(0, min(len(eval_chunks), timed_batches * batch_size), batch_size):
        chunk_group = eval_chunks[i : i + batch_size]
        inps = torch.stack([x["input_ids"] for x in chunk_group]).to(device)
        masks = torch.stack([x["attention_mask"] for x in chunk_group]).to(device)
        batches.append((inps, masks))

    if not batches:
        return 0.0, 0.0

    # Warmup
    with torch.no_grad():
        for b_inps, b_masks in batches[:2]:
            model(input_ids=b_inps, attention_mask=b_masks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed runs
    total_tokens = sum(b[0].numel() for b in batches)
    t0 = time.perf_counter()
    with torch.no_grad():
        for b_inps, b_masks in batches:
            model(input_ids=b_inps, attention_mask=b_masks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / len(batches)) * 1000.0 if batches else 0.0
    return round(throughput, 2), round(latency_ms, 2)


def verify_checkpoint_integrity(
    model: nn.Module,
    tokenizer: Any,
    ckpt_dir: Path,
    device: str,
    val_input_ids: torch.Tensor,
    val_attention_mask: torch.Tensor,
) -> Dict[str, Any]:
    """
    Strictly verifies checkpoint serialization and reload without ignore_mismatched_sizes.
    Passes identical attention_mask to both reference and reloaded models.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    with torch.no_grad():
        ref_logits = model(input_ids=val_input_ids, attention_mask=val_attention_mask).logits

    # Reload with strict loading
    reloaded, info = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=model.dtype,
        device_map=device if device != "cpu" else None,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
    )
    reloaded.eval()

    missing = list(info.get("missing_keys", []))
    unexpected = list(info.get("unexpected_keys", []))
    mismatched = list(info.get("mismatched_keys", []))

    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=val_input_ids, attention_mask=val_attention_mask).logits

    diff = (ref_logits.float() - reloaded_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    del reloaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    passed = (len(missing) == 0 and len(unexpected) == 0 and len(mismatched) == 0 and max_diff < 1e-2)
    return {
        "status": "PASS" if passed else "FAIL",
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "mismatched_keys_count": len(mismatched),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "max_logit_diff": round(max_diff, 6),
        "mean_logit_diff": round(mean_diff, 6),
    }



def main():
    p = argparse.ArgumentParser(description="CARE-MoE L40S Benchmark Campaign Orchestrator")
    p.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0924", help="Model ID")
    p.add_argument("--methods", nargs="+", default=["care_static", "care_adaptive", "random"],
                   help="Methods to run (care_static, care_adaptive, random, sub_moe, reap_prune, hc_smoe, mc_smoe)")
    p.add_argument("--targets", nargs="+", type=int, default=[60, 56, 48],
                   help="Target expert counts (e.g. 60 56 48)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate")
    p.add_argument("--eval-chunks", type=int, default=128, help="Number of 512-token eval chunks (128 = 65,536 tokens)")
    p.add_argument("--batch-size", type=int, default=4, help="Eval batch size")
    p.add_argument("--cal-samples", type=int, default=128, help="Calibration sample size")
    p.add_argument("--device", default="cuda:0", help="CUDA device")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--router-merge-mode", default="uniform", choices=["uniform", "frequency_weighted"])
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "BENCHMARK_L40S"))
    p.add_argument("--keep-checkpoints", action="store_true", help="Retain saved checkpoint shards on disk")
    p.add_argument("--reuse-results-dir", default=None, help="Directory containing previously verified run results to reuse")
    args = p.parse_args()

    # Create collision-resistant run directory (never overwrite)
    run_dir = create_unique_run_dir(Path(args.output_dir))

    print(f"\n{'='*75}")
    print(f"  CARE-MoE Benchmark Campaign on NVIDIA L40S")
    print(f"  Run Directory: {run_dir}")
    print(f"  Methods:       {args.methods}")
    print(f"  Targets:       {args.targets}")
    print(f"  Seeds:         {args.seeds}")
    print(f"  Evaluation:    WikiText-2 test ({args.eval_chunks} chunks x 512 = {args.eval_chunks * 512} tokens)")
    print(f"{'='*75}\n")

    # Step 1: Strictly Validate and Load Calibration Data (BEFORE loading any large model)
    parquet_path = PROJECT_ROOT / "results" / "exp6c" / "token_vectors" / "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    print(f"[CALIBRATION CHECK] Validating required calibration artifact: {parquet_path}...")
    df_cal_raw, cal_meta = load_and_validate_calibration_data(parquet_path, required_samples=args.cal_samples)
    print(f"[CALIBRATION CHECK] Artifact validated successfully: {cal_meta['total_rows']} rows, SHA-256={cal_meta['sha256'][:16]}...")

    # Step 2: Build Planned Combinations Matrix
    planned_combinations: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for method in args.methods:
            for target in sorted(args.targets, reverse=True):
                planned_combinations.append({
                    "method": method,
                    "target_experts": target,
                    "seed": seed,
                })
    print(f"[PLANNING] Total planned execution runs: {len(planned_combinations)}")

    # Step 3: Environment, Hardware, Git, and Initial Manifest
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    git_info = get_git_info()
    source_hashes = get_critical_source_hashes()

    manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "campaign_status": "IN_PROGRESS",
        "git": git_info,
        "critical_source_hashes": source_hashes,
        "calibration_artifact": cal_meta,
        "cli_arguments": vars(args),
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "evaluation_settings": {
            "dataset": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "test_split": "test",
            "validation_split": "validation",
            "eval_token_count": args.eval_chunks * 512,
            "eval_chunks": args.eval_chunks,
            "eval_batch_size": args.batch_size,
            "max_length": 512,
            "heldout_kl_chunks": 16,
            "causally_aligned_kl": True,
        },
        "candidate_budget": {
            "candidate_k": 5,
            "candidate_pool_size": 5,
        },
        "router_configuration": {
            "router_merge_mode": args.router_merge_mode,
            "renormalize_router_weights": True,
        },
        "methods": args.methods,
        "targets": args.targets,
        "seeds": args.seeds,
        "planned_combinations": planned_combinations,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Step 4: Tokenizer and Data Preparation
    print("[INIT] Loading tokenizer and evaluation data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_eval_config = {
        "evaluation": {
            "dataset": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "test",
            "max_length": 512,
            "num_eval_chunks": args.eval_chunks,
        }
    }
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, test_eval_config)

    selection_config = {
        "evaluation": {
            "dataset": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "validation",
            "max_length": 512,
            "num_eval_chunks": 16,
        }
    }
    selection_batches = prepare_wikitext_eval_batches(tokenizer, selection_config)

    # Assert strictly zero overlap between selection data and test evaluation data
    eval_token_hashes = {hashlib.sha256(x["input_ids"].numpy().tobytes()).hexdigest() for x in eval_chunks}
    sel_token_hashes = {hashlib.sha256(x["input_ids"].numpy().tobytes()).hexdigest() for x in selection_batches}
    overlap = eval_token_hashes.intersection(sel_token_hashes)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} chunks overlap between selection and test sets!"
    print(f"[DATA INTEGRITY] Selection data (validation split: {len(selection_batches)} chunks) and Final PPL data (test split: {len(eval_chunks)} chunks) verified 100% disjoint (0 chunks overlap).")

    # Step 5: Model Loading & Reference Baseline Measurement
    print(f"[MODEL] Loading reference model {args.model_id} in {args.dtype}...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    # Record model and tokenizer revisions
    model_rev = getattr(base_model.config, "_commit_hash", None) or getattr(base_model.config, "transformers_version", "unknown")
    tokenizer_rev = getattr(tokenizer, "_commit_hash", None) or "unknown"
    manifest["model_revision"] = str(model_rev)
    manifest["tokenizer_revision"] = str(tokenizer_rev)

    base_params = count_parameters(base_model)
    adapter = OLMoEModelAdapter()
    moe_blocks = adapter.get_moe_blocks(base_model)
    initial_experts = adapter.get_num_experts(moe_blocks[0][1])

    print(f"[BASELINE] Measuring Reference PPL (64 Experts, {base_params:,} parameters)...")
    base_ppl = compute_ppl(base_model, eval_chunks, device=args.device, batch_size=args.batch_size)
    print(f"[BASELINE] Reference PPL: {base_ppl:.4f}")

    print("[BASELINE] Measuring Reference Inference Speed...")
    base_throughput, base_latency = measure_inference_speed(
        base_model, eval_chunks, device=args.device, batch_size=args.batch_size
    )
    base_peak_vram = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0

    print(f"[BASELINE] Throughput: {base_throughput} tokens/s | Latency: {base_latency} ms/batch | VRAM: {base_peak_vram} MB")

    print("[BASELINE] Precomputing held-out test baseline logits (16 chunks x 512 = 8,192 tokens)...")
    baseline_test_logits = []
    with torch.no_grad():
        for b_idx in range(0, min(len(eval_chunks), 16), args.batch_size):
            b = eval_chunks[b_idx : b_idx + args.batch_size]
            b_in = torch.stack([x["input_ids"] for x in b]).to(args.device)
            b_mask = torch.stack([x["attention_mask"] for x in b]).to(args.device)
            l = base_model(input_ids=b_in, attention_mask=b_mask).logits.cpu()
            baseline_test_logits.append(l)

    baseline_record = {
        "experts": initial_experts,
        "parameters": base_params,
        "perplexity": round(base_ppl, 4),
        "throughput_tokens_per_s": base_throughput,
        "latency_ms_per_batch": base_latency,
        "peak_vram_mb": base_peak_vram,
    }
    with open(run_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_record, f, indent=2)

    # Precompute baseline probe distributions for all seeds while base_model is on GPU
    seed_baseline_probs = {}
    for seed in args.seeds:
        df_cal = df_cal_raw.sample(n=min(args.cal_samples, len(df_cal_raw)), random_state=seed).reset_index(drop=True)
        print(f"[CALIBRATION] Precomputing baseline calibration distribution (seed={seed})...")
        seed_baseline_probs[seed] = compute_calibration_distributions(
            base_model, df_cal, batch_size=32, device=args.device
        )

    # Free base_model from GPU to ensure maximum VRAM headroom during compression runs
    print("[INIT] Moving base model to CPU to eliminate GPU VRAM contention...")
    base_model = base_model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 6: Multi-Target Compression Execution Loop with Strict Run Tracking
    all_results: List[Dict[str, Any]] = []

    for combo in planned_combinations:
        method = combo["method"]
        target = combo["target_experts"]
        seed = combo["seed"]
        combo_key = f"{method}_{target}e_seed_{seed}"

        if args.reuse_results_dir:
            reuse_file = Path(args.reuse_results_dir) / f"{combo_key}_result.json"
            if reuse_file.exists():
                try:
                    with open(reuse_file, "r") as rf:
                        reused = json.load(rf)
                    if (reused.get("status") == "SUCCESS" and 
                        reused.get("checkpoint_integrity", {}).get("status") == "PASS" and 
                        reused.get("all_layers_verified") is True):
                        print(f"\n{'='*65}")
                        print(f"  [REUSE] Verified result found: {combo_key} in {reuse_file}")
                        print(f"  PPL: {reused.get('compressed_ppl')} | Delta PPL: {reused.get('ppl_degradation')} | Reload: PASS")
                        print(f"{'='*65}")
                        with open(run_dir / f"{combo_key}_result.json", "w") as out_f:
                            json.dump(reused, out_f, indent=2)
                        all_results.append(reused)
                        continue
                except Exception as r_err:
                    print(f"[REUSE WARNING] Could not reuse {reuse_file}: {r_err}. Executing fresh.")

        print(f"\n{'='*65}")
        print(f"  RUNNING: {method.upper()} | Target: {target} experts | Seed: {seed}")
        print(f"{'='*65}")

        df_cal = df_cal_raw.sample(n=min(args.cal_samples, len(df_cal_raw)), random_state=seed).reset_index(drop=True)
        baseline_probs = seed_baseline_probs[seed]

        # Initialize run record with FAILED status until all validation gates pass
        run_record: Dict[str, Any] = {
            "method": method,
            "display_name": method,
            "target_experts": target,
            "actual_experts": 0,
            "seed": seed,
            "status": "FAILED",
            "failure_reason": None,
            "traceback": None,
            "all_layers_verified": False,
            "all_layer_expert_counts": [],
            "fidelity_status": "UNKNOWN",
            "parameters": 0,
            "param_reduction_pct": 0.0,
            "baseline_ppl": round(base_ppl, 4),
            "compressed_ppl": None,
            "ppl_degradation": None,
            "heldout_test_kl": None,
            "probe_functional_kl": None,
            "stepwise_marginal_kl_sum": None,
            "compression_time_s": 0.0,
            "peak_vram_mb": 0.0,
            "throughput_tokens_per_s": 0.0,
            "latency_ms_per_batch": 0.0,
            "checkpoint_integrity": None,
            "warnings": [],
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model_copy = copy.deepcopy(base_model).to(args.device).eval()

        compressor_config = {
            "router_merge_mode": args.router_merge_mode,
            "candidate_pool_size": 5,
            "candidate_k": 5,
            "num_axes": 10,
            "samples_per_axis": 20,
            "eval_batches": selection_batches,
            "max_cal_batches": 16,
            "metric": "reap",
            "rank_ratio": 0.5,
            "renormalize_router_weights": True,
        }

        try:
            compressor = get_compressor(method, config=compressor_config, model_adapter=adapter)
        except Exception as comp_init_err:
            print(f"[ERROR] Compressor initialization failed for {method}: {comp_init_err}")
            run_record["failure_reason"] = f"COMPRESSOR_INIT_FAILED: {comp_init_err}"
            run_record["traceback"] = traceback.format_exc()
            with open(run_dir / f"{combo_key}_result.json", "w") as f:
                json.dump(run_record, f, indent=2)
            all_results.append(run_record)
            del model_copy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if method in ["reap_prune", "reap", "sub_moe", "sub-moe"]:
            cal_input = selection_batches
        elif method in ["care_static", "care_adaptive", "care_static_k1", "care_static_k5", "care_adaptive_k5"]:
            cal_input = df_cal
        elif method in ["random", "random_control"]:
            cal_input = selection_batches
        else:
            cal_input = df_cal

        # Compression execution
        t_comp_start = time.perf_counter()
        try:
            res: CompressionResult = compressor.compress(
                model=model_copy,
                calibration_data=cal_input,
                target_experts=target,
                seed=seed,
            )
            comp_time = time.perf_counter() - t_comp_start
        except Exception as comp_err:
            print(f"[ERROR] Compression failed for {method} @ {target}E: {comp_err}")
            run_record["failure_reason"] = f"COMPRESSION_FAILED: {comp_err}"
            run_record["traceback"] = traceback.format_exc()
            with open(run_dir / f"{combo_key}_result.json", "w") as f:
                json.dump(run_record, f, indent=2)
            all_results.append(run_record)
            del model_copy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        comp_peak_vram = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0

        # Expert count verification across all layers
        compressed_blocks = adapter.get_moe_blocks(res.model)
        actual_counts = [adapter.get_num_experts(b[1]) for b in compressed_blocks]
        all_layers_matched = (len(actual_counts) > 0 and all(c == target for c in actual_counts))
        run_record["actual_experts"] = actual_counts[0] if actual_counts else 0
        run_record["all_layer_expert_counts"] = actual_counts
        run_record["all_layers_verified"] = all_layers_matched
        run_record["display_name"] = res.method_id
        run_record["fidelity_status"] = res.fidelity_status
        run_record["warnings"] = res.warnings

        if not all_layers_matched:
            err_msg = (
                f"EXPERT_COUNT_MISMATCH: expected {target} across all {len(compressed_blocks)} layers, "
                f"got counts: {actual_counts}"
            )
            print(f"[ERROR] {err_msg}")
            run_record["failure_reason"] = err_msg
            with open(run_dir / f"{combo_key}_result.json", "w") as f:
                json.dump(run_record, f, indent=2)
            all_results.append(run_record)
            del model_copy
            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        print(f"[{method.upper()}] Layers verified: {actual_counts[:4]}... All {len(actual_counts)} layers = {target} (MATCHED)")

        # Full Evaluation Suite inside exception guard
        try:
            # 1. Measure compressed perplexity on held-out test split
            print(f"[{method.upper()}] Evaluating Perplexity on WikiText-2 (test split)...")
            comp_ppl = compute_ppl(res.model, eval_chunks, device=args.device, batch_size=args.batch_size)
            ppl_delta = round(comp_ppl - base_ppl, 4)
            print(f"[{method.upper()}] PPL: {comp_ppl:.4f} (Δ={ppl_delta:+.4f})")

            # 2. Causally Aligned Held-out Test KL Divergence
            print(f"[{method.upper()}] Evaluating Causally Aligned Held-out Test KL against Reference Baseline...")
            heldout_kl_val = 0.0
            heldout_tokens = 0
            with torch.no_grad():
                for b_idx in range(0, min(len(eval_chunks), 16), args.batch_size):
                    b = eval_chunks[b_idx : b_idx + args.batch_size]
                    b_in = torch.stack([x["input_ids"] for x in b]).to(args.device)
                    b_mask = torch.stack([x["attention_mask"] for x in b]).to(args.device)
                    comp_l = res.model(input_ids=b_in, attention_mask=b_mask).logits
                    base_l = baseline_test_logits[b_idx // args.batch_size].to(args.device)

                    batch_kl, batch_tokens = compute_causal_kl_from_logits(base_l, comp_l, b_mask)
                    heldout_kl_val += batch_kl
                    heldout_tokens += batch_tokens

            if heldout_tokens == 0:
                raise ValueError("Zero valid target tokens found during held-out causal KL evaluation.")
            heldout_test_kl = heldout_kl_val / heldout_tokens
            print(f"[{method.upper()}] Causally Aligned Held-out Test KL: {heldout_test_kl:.6f} ({heldout_tokens:,} tokens)")

            # 3. Measure probe functional KL
            print(f"[{method.upper()}] Evaluating Probe Functional KL Divergence...")
            probe_kl = evaluate_marginal_kl(res.model, df_cal, baseline_probs, batch_size=32, device=args.device)
            print(f"[{method.upper()}] Probe Functional KL: {probe_kl:.6f}")

            # 4. Stepwise marginal KL sum
            step_records = getattr(res, "trajectory", None) or getattr(res, "step_records", None) or []
            marginal_damages = [float(r["marginal_damage"]) for r in step_records if "marginal_damage" in r]
            marginal_kl_sum = sum(marginal_damages) if marginal_damages else None

            # 5. Measure compressed inference speed
            print(f"[{method.upper()}] Measuring Inference Speed...")
            comp_throughput, comp_latency = measure_inference_speed(
                res.model, eval_chunks, device=args.device, batch_size=args.batch_size
            )
            comp_params = count_parameters(res.model)

            # 6. Strict Checkpoint Reload Parity Test (with attention mask)
            print(f"[{method.upper()}] Testing Strict Checkpoint Serialization & Reload with Attention Mask...")
            ckpt_dir = run_dir / "checkpoints" / combo_key
            val_input = torch.stack([x["input_ids"] for x in eval_chunks[:1]]).to(args.device)
            val_mask = torch.stack([x["attention_mask"] for x in eval_chunks[:1]]).to(args.device)

            ckpt_audit = verify_checkpoint_integrity(
                res.model, tokenizer, ckpt_dir, args.device, val_input, val_mask
            )
            print(f"[{method.upper()}] Checkpoint Reload: {ckpt_audit['status']} | Missing={ckpt_audit['missing_keys_count']} Mismatched={ckpt_audit['mismatched_keys_count']} MaxLogitDiff={ckpt_audit['max_logit_diff']}")

            if not args.keep_checkpoints and ckpt_dir.exists():
                shutil.rmtree(ckpt_dir, ignore_errors=True)

            # Check if reload parity failed
            if ckpt_audit["status"] != "PASS":
                err_msg = (
                    f"CHECKPOINT_RELOAD_FAILED: status={ckpt_audit['status']}, "
                    f"missing_keys={ckpt_audit['missing_keys_count']}, "
                    f"mismatched_keys={ckpt_audit['mismatched_keys_count']}, "
                    f"max_logit_diff={ckpt_audit['max_logit_diff']}"
                )
                print(f"[ERROR] {err_msg}")
                run_record["status"] = "FAILED"
                run_record["failure_reason"] = err_msg
            else:
                run_record["status"] = "SUCCESS"

            # Populate evaluation metrics
            run_record["parameters"] = comp_params
            run_record["param_reduction_pct"] = round((1.0 - comp_params / base_params) * 100.0, 2)
            run_record["compressed_ppl"] = round(comp_ppl, 4)
            run_record["ppl_degradation"] = ppl_delta
            run_record["heldout_test_kl"] = round(float(heldout_test_kl), 6)
            run_record["probe_functional_kl"] = round(float(probe_kl), 6)
            run_record["stepwise_marginal_kl_sum"] = round(float(marginal_kl_sum), 6) if marginal_kl_sum is not None else None
            run_record["compression_time_s"] = round(comp_time, 2)
            run_record["peak_vram_mb"] = comp_peak_vram
            run_record["throughput_tokens_per_s"] = comp_throughput
            run_record["latency_ms_per_batch"] = comp_latency
            run_record["checkpoint_integrity"] = ckpt_audit

        except Exception as eval_err:
            print(f"[ERROR] Post-compression evaluation failed for {method} @ {target}E: {eval_err}")
            run_record["status"] = "FAILED"
            run_record["failure_reason"] = f"EVALUATION_FAILED: {eval_err}"
            run_record["traceback"] = traceback.format_exc()

        # Save individual run artifact
        with open(run_dir / f"{combo_key}_result.json", "w") as f:
            json.dump(run_record, f, indent=2)
        all_results.append(run_record)

        del model_copy
        del res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 7: Final Campaign Accounting and Status Resolution
    successful_runs = [r for r in all_results if r.get("status") == "SUCCESS"]
    failed_runs = [r for r in all_results if r.get("status") != "SUCCESS"]
    all_planned_keys = {f"{c['method']}_{c['target_experts']}e_seed_{c['seed']}" for c in planned_combinations}
    recorded_keys = {f"{r['method']}_{r['target_experts']}e_seed_{r['seed']}" for r in all_results}
    missing_keys = all_planned_keys - recorded_keys

    campaign_status = "COMPLETE" if (len(failed_runs) == 0 and len(missing_keys) == 0 and len(all_results) == len(planned_combinations)) else "INCOMPLETE"

    print(f"\n{'='*75}")
    print(f"  CAMPAIGN SUMMARY: {campaign_status}")
    print(f"  Planned Runs:    {len(planned_combinations)}")
    print(f"  Successful Runs: {len(successful_runs)}")
    print(f"  Failed Runs:     {len(failed_runs)}")
    print(f"  Missing Runs:    {len(missing_keys)}")
    print(f"{'='*75}\n")

    manifest["campaign_status"] = campaign_status
    manifest["planned_runs_count"] = len(planned_combinations)
    manifest["successful_runs_count"] = len(successful_runs)
    manifest["failed_runs_count"] = len(failed_runs)
    manifest["missing_runs_count"] = len(missing_keys)
    manifest["missing_combinations"] = list(missing_keys)
    manifest["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Write final manifest
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save consolidated summary JSON
    with open(run_dir / "benchmark_summary.json", "w") as f:
        json.dump({
            "manifest": manifest,
            "baseline": baseline_record,
            "planned_combinations": planned_combinations,
            "results": all_results,
            "failed_runs": failed_runs,
        }, f, indent=2)

    # Step 8: Generate Rigorous Markdown Report
    status_badge = "✅ **COMPLETE**" if campaign_status == "COMPLETE" else "❌ **INCOMPLETE**"
    report_lines = [
        "# CARE-MoE Controlled Benchmark Campaign Report (NVIDIA L40S)",
        f"**Campaign Status:** {status_badge}  ",
        f"**Run Timestamp:** `{manifest['timestamp']}`  ",
        f"**GPU:** {manifest['gpu_name']} ({manifest['gpu_total_memory_gb']} GB VRAM)  ",
        f"**Model:** `{manifest['model_id']}` (OLMoE-1B-7B) | **Dtype:** `{manifest['dtype']}`  ",
        f"**Evaluation Corpus:** WikiText-2 `test` split ({manifest['evaluation_settings']['eval_token_count']:,} tokens across 128 chunks of 512)  ",
        f"**Held-out KL:** Causally aligned (`logits[:, :-1, :]` vs `attention_mask[:, 1:]`) over 16 chunks (8,192 tokens)  ",
        f"**Router Merge Mode:** `{manifest['router_configuration']['router_merge_mode']}`  ",
        f"**Git Revision:** `{manifest['git']['revision'][:12]}` (Dirty: `{manifest['git']['is_dirty']}`)  ",
        f"**Calibration Parquet SHA-256:** `{manifest['calibration_artifact']['sha256'][:16]}...` ({manifest['calibration_artifact']['total_rows']} rows)  ",
        "",
        "---",
        "",
    ]

    if campaign_status != "COMPLETE":
        report_lines.extend([
            "> [!CAUTION]",
            "> **CAMPAIGN INCOMPLETE**: One or more planned benchmark runs failed or were omitted.",
            f"> Planned: {len(planned_combinations)} | Succeeded: {len(successful_runs)} | Failed: {len(failed_runs)} | Missing: {len(missing_keys)}",
            "> Review the failed runs section below for root-cause tracebacks and diagnostics.",
            "",
        ])

    report_lines.extend([
        "## 1. Reference Baseline (64 Experts)",
        f"- **Parameters:** {baseline_record['parameters']:,}",
        f"- **Perplexity (PPL):** **{baseline_record['perplexity']}**",
        f"- **Inference Throughput:** {baseline_record['throughput_tokens_per_s']} tokens/s",
        f"- **Inference Latency:** {baseline_record['latency_ms_per_batch']} ms/batch (batch_size={args.batch_size})",
        f"- **Peak VRAM:** {baseline_record['peak_vram_mb']} MB",
        "",
        "---",
        "",
        "## 2. Benchmark Results",
        "",
        "| Method | Target | Verified Experts | Status | PPL | Δ PPL | Causal Held-out KL | Probe KL | Marginal KL Sum | Time (s) | Reload |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r in all_results:
        status_icon = "✅ OK" if r.get("status") == "SUCCESS" else "❌ FAILED"
        reload_str = "-"
        if r.get("checkpoint_integrity"):
            reload_str = "✅ PASS" if r["checkpoint_integrity"]["status"] == "PASS" else "❌ FAIL"
        
        ppl_str = f"**{r['compressed_ppl']}**" if r.get("compressed_ppl") is not None else "-"
        ppl_delta_str = f"{r['ppl_degradation']:+.4f}" if r.get("ppl_degradation") is not None else "-"
        heldout_kl_str = f"{r['heldout_test_kl']:.6f}" if r.get("heldout_test_kl") is not None else "-"
        probe_kl_str = f"{r['probe_functional_kl']:.6f}" if r.get("probe_functional_kl") is not None else "-"
        m_kl_str = f"{r['stepwise_marginal_kl_sum']:.6f}" if r.get("stepwise_marginal_kl_sum") is not None else "-"
        comp_time_str = f"{r['compression_time_s']}s" if r.get("compression_time_s", 0) > 0 else "-"
        exp_match_icon = "✅" if r.get("all_layers_verified") else "❌"

        report_lines.append(
            f"| **{r['method']}** (s={r['seed']}) | {r['target_experts']}E | "
            f"{r['actual_experts']}E ({exp_match_icon}) | {status_icon} | "
            f"{ppl_str} | {ppl_delta_str} | {heldout_kl_str} | "
            f"{probe_kl_str} | {m_kl_str} | {comp_time_str} | {reload_str} |"
        )

    if failed_runs:
        report_lines.extend([
            "",
            "---",
            "",
            "## 3. Failed Run Diagnostics",
            "",
        ])
        for fr in failed_runs:
            report_lines.extend([
                f"### ❌ `{fr['method']}` @ {fr['target_experts']}E (Seed {fr['seed']})",
                f"- **Failure Reason:** `{fr.get('failure_reason')}`",
            ])
            if fr.get("traceback"):
                report_lines.extend([
                    "```",
                    fr["traceback"].strip(),
                    "```",
                ])

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Scientific Fidelity and Methodological Status",
        "",
        "- **CARE-Static (`FAITHFUL`):** Single frozen initial capability ranking, explicit active-to-original identity tracking.",
        "- **CARE-Adaptive (`FAITHFUL`):** Dynamic capability recomputation at each step, top-5 candidate pool, transactional argmin KL damage evaluation.",
        "- **Random Control (`CONTROL`):** Matched K=5 candidate pool evaluated physically via marginal KL.",
        "- **Sub-MoE (`FAITHFUL`):** Joint SVD decomposition of expert weight matrices with frequency-weighted singular component reconstruction.",
        "- **REAP-Prune (`FAITHFUL`):** Router-weighted activation norm saliency accumulation and physical expert slice deletion.",
        "- **HC-SMoE (`PARTIAL (router-proxy)`):** Approximated using router weight cosine similarity rather than calibration activation profiles.",
        "- **MC-SMoE (`PARTIAL (Stages 1–4 only)`):** Implements Hungarian bipartite neuron matching and permuted merge; Stage 5 (low-rank SVD residual pruning) is omitted.",
        "",
        f"*Generated automatically by benchmark/run_l40s_benchmark.py at {datetime.datetime.now(datetime.timezone.utc).isoformat()}*",
    ])

    report_path = run_dir / "BENCHMARK_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n[COMPLETE] Campaign finished with status: {campaign_status}. Report: {report_path}")


if __name__ == "__main__":
    main()
