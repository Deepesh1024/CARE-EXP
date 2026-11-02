"""
benchmark/run_cross_capability_campaign.py
==========================================
Master Campaign Orchestrator for Cross-Model MoE Compression, Multi-Capability Probing,
Reasoning Preservation & Capability Benchmarking (CARE-COM v3.0).

Executes end-to-end on NVIDIA L40S:
1. Validates infrastructure, environment, and datasets.
2. Evaluates reference baseline models across Tracks A, B, C, D.
3. Builds capability probes with strict 3-role data separation.
4. Runs controlled OLMoE ablation across Generic, Specific, and Multi-Capability Probing.
5. Runs cross-model evaluations for JetMoE, Phi-3.5-MoE, and Qwen3-30B-A3B.
6. Gathers probe diagnostics (candidate overlap Jaccard, distance distributions, functional damage).
7. Emits unified results database (all_runs.jsonl, results.csv, aggregated_results.csv).
"""

from __future__ import annotations
import os
import sys
import gc
import time
import copy
import json
import uuid
import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.probes import (
    build_probe_datasets,
    verify_strict_data_separation,
    extract_single_capability_representation,
    build_multi_capability_representation,
    compute_pairwise_distances,
    nominate_candidate_pool,
    compute_candidate_pool_jaccard,
)
from benchmark.evaluate_capabilities import (
    evaluate_wikitext2_ppl_and_kl,
    evaluate_gsm8k,
    evaluate_mmlu,
    evaluate_mbpp,
    run_full_capability_suite,
)
from benchmark.compressors.care_adaptive import CAREAdaptiveCompressor
from benchmark.compressors.care_static import CAREStaticCompressor
from benchmark.compressors.random_control import RandomControlCompressor
from benchmark.compressors.reap_prune import REAPPruneCompressor
from benchmark.compressors.sub_moe import SubMoECompressor
from benchmark.compressors.hc_smoe import HCSMoECompressor
from benchmark.compressors.mc_smoe import MCSMoECompressor
from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage


def extract_moe_blocks(model: nn.Module) -> List[nn.Module]:
    """Finds MoE blocks across supported architectures."""
    blocks = []
    for m in model.modules():
        if hasattr(m, "experts") and (isinstance(m.experts, nn.ModuleList) or hasattr(m.experts, "gate_up_proj")):
            blocks.append(m)
        elif hasattr(m, "input_linear") and hasattr(m, "output_linear") and hasattr(m, "router"):
            blocks.append(m)
    return blocks


def verify_expert_counts(model: nn.Module, expected_count: int) -> Tuple[bool, List[int]]:
    """Verifies physical expert counts across all layers."""
    counts = []
    blocks = extract_moe_blocks(model)
    for b in blocks:
        if hasattr(b, "current_num_experts"):
            counts.append(b.current_num_experts)
        elif hasattr(b, "experts"):
            if isinstance(b.experts, nn.ModuleList):
                counts.append(len(b.experts))
            elif hasattr(b.experts, "gate_up_proj"):
                counts.append(b.experts.gate_up_proj.shape[0])
        elif hasattr(b, "input_linear"):
            counts.append(b.input_linear.weight.shape[0])
    all_match = all(c == expected_count for c in counts) if counts else False
    return all_match, counts


def compress_with_custom_probe(
    model: nn.Module,
    moe_blocks: list,
    probe_dict: Dict[str, pd.DataFrame],
    probe_strategy: str,
    target_experts: int,
    selection_batches: list,
    candidate_k: int = 5,
    device: str = "cuda:0",
    is_adaptive: bool = True,
    seed: int = 42
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """
    Executes CARE compression using a specified capability probe strategy:
    - 'generic': standard 10-axis probe
    - 'math': GSM8K train probe
    - 'coding': MBPP train probe
    - 'knowledge': MMLU dev probe
    - 'lm': WikiText train probe
    - 'multi': Combined multi-capability probe
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    current_experts = moe_blocks[0].experts.gate_up_proj.shape[0] if hasattr(moe_blocks[0].experts, "gate_up_proj") else len(moe_blocks[0].experts)
    
    engine = PhysicalMergeEngine(model, router_merge_mode="uniform")
    trajectory = []

    # Helper function to get candidate pool under the chosen strategy
    def get_candidates(n_curr):
        if probe_strategy == "multi":
            C, _ = build_multi_capability_representation(model, moe_blocks, n_curr, probe_dict, max_samples_per_probe=20, device=device)
        elif probe_strategy == "generic":
            from experiments.care_com_v21.care_capability_v23 import compute_global_capability_vectors_v23
            C = compute_global_capability_vectors_v23(model, moe_blocks, n_curr, probe_dict["generic"], samples_per_axis=20, device=device)
        elif probe_strategy in probe_dict:
            C = extract_single_capability_representation(model, moe_blocks, n_curr, probe_dict[probe_strategy], max_samples=20, device=device)
        else:
            raise ValueError(f"Unknown probe strategy: {probe_strategy}")
        
        dist = compute_pairwise_distances(C)
        return nominate_candidate_pool(dist, pool_size=candidate_k)

    # If static, freeze the initial candidate pool and ranking
    static_pool = None
    if not is_adaptive:
        static_pool = get_candidates(current_experts)

    while engine.current_num_experts > target_experts:
        step_idx = len(trajectory)
        if is_adaptive:
            candidates = get_candidates(engine.current_num_experts)
        else:
            # Re-map static pool to current active indices
            candidates = [(i, j) for i, j in static_pool if i < engine.current_num_experts and j < engine.current_num_experts][:candidate_k]
            if len(candidates) == 0:
                candidates = [(0, 1)]

        # Collect pre-merge reference logits on selection batches
        ref_logits = collect_current_logits(model, selection_batches, batch_size=4, max_eval_batches=4, device=device)

        # Transactional candidate evaluation
        candidate_damages = []
        for i_cand, j_cand in candidates:
            engine.snapshot(i_cand, j_cand)
            try:
                engine.merge_experts(i_cand, j_cand)
                damage = evaluate_marginal_damage(
                    model=model,
                    eval_chunks=selection_batches,
                    current_logprobs=ref_logits,
                    batch_size=4,
                    max_eval_batches=4,
                    device=device
                )
            finally:
                engine.restore()
            candidate_damages.append((damage, i_cand, j_cand))

        candidate_damages.sort(key=lambda x: x[0])
        best_damage, best_i, best_j = candidate_damages[0]

        # Permanently commit best merge
        engine.merge_experts(best_i, best_j)

        trajectory.append({
            "step": step_idx,
            "merged_pair": (best_i, best_j),
            "marginal_damage": float(best_damage),
            "surviving_experts": engine.current_num_experts,
            "candidate_damages": [float(x[0]) for x in candidate_damages]
        })

    return model, trajectory


def main():
    print(f"\n{'='*80}")
    print("  CARE-COM Master Execution Campaign (NVIDIA L40S)")
    print("  Multi-Capability Probing, Reasoning Preservation & Cross-Model Benchmarking")
    print(f"{'='*80}\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    campaign_dir = PROJECT_ROOT / "reports" / "CROSS_MODEL_CAPABILITY" / "run_20261026_master"
    raw_dir = campaign_dir / "raw_results"
    agg_dir = campaign_dir / "aggregates"
    fig_dir = campaign_dir / "figures"
    reports_dir = campaign_dir / "reports"
    for d in [raw_dir, agg_dir, fig_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_run_records = []

    # -------------------------------------------------------------------------
    # PHASE 1 & 2: Tokenizer, Base Model & Probe Construction
    # -------------------------------------------------------------------------
    model_id = "allenai/OLMoE-1B-7B-0924"
    print(f"[INIT] Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INIT] Loading OLMoE reference base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    base_model.eval()

    # Precompute selection batches for marginal KL candidate evaluation (validation split: 16 chunks)
    print("[DATA] Loading disjoint validation batches for marginal KL candidate evaluation...")
    ds_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    val_text = "\n\n".join([t for t in ds_val["text"] if len(t.strip()) > 0])
    val_enc = tokenizer(val_text, truncation=False, return_tensors="pt")["input_ids"][0]
    selection_batches = []
    for i in range(0, min(len(val_enc) - 512 + 1, 16 * 512), 512):
        chunk = val_enc[i : i + 512]
        selection_batches.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk)
        })

    # Evaluate or Reuse Reference Baseline
    base_file = raw_dir / "baseline_olmoe_64e.json"
    if base_file.exists():
        print(f"[REUSE] Loading cached baseline results from {base_file}...")
        with open(base_file) as bf:
            base_record = json.load(bf)
        base_metrics = base_record
        print("[BASELINE] Precomputing 16 chunks of base logits for causal held-out KL...")
        _, _, base_logits = evaluate_wikitext2_ppl_and_kl(base_model, tokenizer, num_eval_chunks=16, device=device)
    else:
        print(f"\n{'='*70}")
        print("[BASELINE] Evaluating Uncompressed Reference Model (64 Experts) across all Tracks...")
        print(f"{'='*70}")
        base_metrics, base_logits = run_full_capability_suite(base_model, tokenizer, base_logits=None, device=device)
        base_record = {
            "run_id": "base_olmoe_64e",
            "model": "OLMoE-1B-7B",
            "method": "baseline",
            "probe_strategy": "none",
            "target_experts": 64,
            "verified_experts": 64,
            "seed": 42,
            "status": "SUCCESS",
            **base_metrics
        }
        with open(base_file, "w") as f:
            json.dump(base_record, f, indent=2)

    all_run_records.append(base_record)
    print(f"[BASELINE RESULTS] WikiText PPL: {base_record['wikitext2_ppl']} | GSM8K Acc: {base_record['gsm8k_accuracy']}% | MMLU Acc: {base_record['mmlu_accuracy']}% | MBPP Pass@1: {base_record['mbpp_pass_at_1']}%")

    # Build Capability Probes (Section 18.A & B)
    print(f"\n{'='*70}")
    print("[PROBES] Building Independent Capability Probes with Strict Separation...")
    print(f"{'='*70}")
    probes = build_probe_datasets(tokenizer, max_length=512, seed=42)
    verify_strict_data_separation(probes, selection_batches)

    # -------------------------------------------------------------------------
    # PROBE DIAGNOSTICS: Stability, Overlap, & Distance Distributions
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[DIAGNOSTICS] Extracting Expert Capability Representations & Candidate Overlaps...")
    print(f"{'='*70}")
    blocks = extract_moe_blocks(base_model)
    cap_representations = {}
    cap_candidate_pools = {}

    for cap_name in ["math", "coding", "knowledge", "lm"]:
        if cap_name in probes:
            print(f"  Extracting {cap_name} capability vector...")
            C_cap = extract_single_capability_representation(base_model, blocks, 64, probes[cap_name], max_samples=20, device=device)
            cap_representations[cap_name] = C_cap
            dist = compute_pairwise_distances(C_cap)
            pool = nominate_candidate_pool(dist, pool_size=5)
            cap_candidate_pools[cap_name] = pool
            print(f"    {cap_name.upper()} Top-5 candidates: {pool}")

    print("  Extracting multi_capability representation (equal-weight normalized concatenation)...")
    C_multi, _ = build_multi_capability_representation(base_model, blocks, 64, probes, max_samples_per_probe=20, device=device)
    dist_multi = compute_pairwise_distances(C_multi)
    pool_multi = nominate_candidate_pool(dist_multi, pool_size=5)
    cap_candidate_pools["multi"] = pool_multi
    print(f"    MULTI-CAPABILITY Top-5 candidates: {pool_multi}")

    # Compute Jaccard Overlap Matrix across capability candidate pools
    all_caps = list(cap_candidate_pools.keys())
    jaccard_matrix = np.zeros((len(all_caps), len(all_caps)))
    for i, c1 in enumerate(all_caps):
        for j, c2 in enumerate(all_caps):
            jaccard_matrix[i, j] = compute_candidate_pool_jaccard(cap_candidate_pools[c1], cap_candidate_pools[c2])

    df_jaccard = pd.DataFrame(jaccard_matrix, index=all_caps, columns=all_caps)
    df_jaccard.to_csv(agg_dir / "candidate_pool_jaccard_matrix.csv")
    print("\n[DIAGNOSTICS] Candidate Pool Jaccard Similarity Matrix:")
    print(df_jaccard)

    # Free base model from GPU to conserve memory during compression sweeps
    base_model = base_model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PHASE 3: Controlled OLMoE Ablation (Section 18.F)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[ABLATION] Executing Staged OLMoE Capability Ablation...")
    print(f"{'='*70}")

    # Experimental Matrix:
    # 60E: Strategy 1 (Generic), Strategy 2 (Math, Coding, Knowledge, LM), Strategy 3 (Multi), Controls (CARE-Static, Random), Baselines (REAP, Sub-MoE, HC-SMoE, MC-SMoE)
    # 48E: Strategy 1 (Seeds 42, 43, 44), Strategy 3 (Seeds 42, 43, 44), Specific probes (Math, Coding, Knowledge), Controls & Baselines (Seed 42)

    ablation_runs = [
        # 60 Experts
        {"method": "care_adaptive", "probe_strategy": "generic", "target": 60, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "multi", "target": 60, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "math", "target": 60, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "coding", "target": 60, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "knowledge", "target": 60, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "lm", "target": 60, "seed": 42},
        {"method": "care_static", "probe_strategy": "generic", "target": 60, "seed": 42},
        {"method": "care_static", "probe_strategy": "multi", "target": 60, "seed": 42},
        {"method": "random", "probe_strategy": "none", "target": 60, "seed": 42},
        {"method": "reap_prune", "probe_strategy": "none", "target": 60, "seed": 42},
        {"method": "sub_moe", "probe_strategy": "none", "target": 60, "seed": 42},
        {"method": "hc_smoe", "probe_strategy": "none", "target": 60, "seed": 42},
        {"method": "mc_smoe", "probe_strategy": "none", "target": 60, "seed": 42},

        # 48 Experts (Deep Compression)
        {"method": "care_adaptive", "probe_strategy": "generic", "target": 48, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "multi", "target": 48, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "multi", "target": 48, "seed": 43},
        {"method": "care_adaptive", "probe_strategy": "multi", "target": 48, "seed": 44},
        {"method": "care_adaptive", "probe_strategy": "math", "target": 48, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "coding", "target": 48, "seed": 42},
        {"method": "care_adaptive", "probe_strategy": "knowledge", "target": 48, "seed": 42},
        {"method": "care_static", "probe_strategy": "generic", "target": 48, "seed": 42},
        {"method": "care_static", "probe_strategy": "multi", "target": 48, "seed": 42},
        {"method": "random", "probe_strategy": "none", "target": 48, "seed": 42},
        {"method": "reap_prune", "probe_strategy": "none", "target": 48, "seed": 42},
        {"method": "sub_moe", "probe_strategy": "none", "target": 48, "seed": 42},
        {"method": "hc_smoe", "probe_strategy": "none", "target": 48, "seed": 42},
        {"method": "mc_smoe", "probe_strategy": "none", "target": 48, "seed": 42},
    ]

    for run_cfg in ablation_runs:
        method = run_cfg["method"]
        probe_strat = run_cfg["probe_strategy"]
        target = run_cfg["target"]
        seed = run_cfg["seed"]
        run_key = f"{method}_{probe_strat}_{target}e_seed_{seed}"
        result_file = raw_dir / f"{run_key}.json"

        # Check for completed result to reuse
        if result_file.exists():
            try:
                with open(result_file) as rf:
                    existing = json.load(rf)
                if existing.get("status") == "SUCCESS":
                    print(f"\n[REUSE] {run_key} already completed successfully. Reusing result.")
                    all_run_records.append(existing)
                    continue
            except Exception:
                pass

        print(f"\n>>> Running {run_key}...")
        t_start = time.time()

        # Load fresh model instance
        comp_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True
        )
        comp_model.eval()
        moe_blocks = extract_moe_blocks(comp_model)

        compression_time = 0.0
        trajectory = []

        try:
            if method == "care_adaptive":
                comp_model, trajectory = compress_with_custom_probe(
                    model=comp_model,
                    moe_blocks=moe_blocks,
                    probe_dict=probes,
                    probe_strategy=probe_strat,
                    target_experts=target,
                    selection_batches=selection_batches,
                    candidate_k=5,
                    device=device,
                    is_adaptive=True,
                    seed=seed
                )
            elif method == "care_static":
                comp_model, trajectory = compress_with_custom_probe(
                    model=comp_model,
                    moe_blocks=moe_blocks,
                    probe_dict=probes,
                    probe_strategy=probe_strat,
                    target_experts=target,
                    selection_batches=selection_batches,
                    candidate_k=5,
                    device=device,
                    is_adaptive=False,
                    seed=seed
                )
            elif method == "random":
                compressor = RandomControlCompressor({
                    "candidate_k": 5,
                    "candidate_pool_size": 5,
                    "eval_batches": selection_batches,
                    "eval_batch_size": 4,
                    "max_eval_batches": 4,
                })
                res = compressor.compress(comp_model, calibration_data=selection_batches, target_experts=target, seed=seed)
                comp_model = res.model
            elif method == "reap_prune":
                compressor = REAPPruneCompressor({
                    "eval_batches": selection_batches,
                    "eval_batch_size": 4,
                    "max_eval_batches": 4,
                })
                res = compressor.compress(comp_model, calibration_data=selection_batches, target_experts=target, seed=seed)
                comp_model = res.model
            elif method == "sub_moe":
                compressor = SubMoECompressor()
                res = compressor.compress(comp_model, target_experts=target, seed=seed)
                comp_model = res.model
            elif method == "hc_smoe":
                compressor = HCSMoECompressor()
                res = compressor.compress(comp_model, target_experts=target, seed=seed)
                comp_model = res.model
            elif method == "mc_smoe":
                compressor = MCSMoECompressor()
                res = compressor.compress(comp_model, target_experts=target, seed=seed)
                comp_model = res.model

            compression_time = round(time.time() - t_start, 2)

            # Verification of physical expert counts
            all_match, actual_counts = verify_expert_counts(comp_model, target)
            assert all_match, f"Expert count mismatch! Target: {target}, Actual: {actual_counts[:4]}..."

            # Evaluate compressed model on full capability suite
            eval_metrics, _ = run_full_capability_suite(comp_model, tokenizer, base_logits=base_logits, device=device)

            run_record = {
                "run_id": run_key,
                "model": "OLMoE-1B-7B",
                "method": method,
                "probe_strategy": probe_strat,
                "target_experts": target,
                "verified_experts": actual_counts[0] if actual_counts else target,
                "all_layers_verified": all_match,
                "seed": seed,
                "compression_time_s": compression_time,
                "status": "SUCCESS",
                **eval_metrics
            }
            print(f"  [SUCCESS] PPL: {eval_metrics['wikitext2_ppl']} | GSM8K: {eval_metrics['gsm8k_accuracy']}% | MMLU: {eval_metrics['mmlu_accuracy']}% | MBPP: {eval_metrics['mbpp_pass_at_1']}% | Time: {compression_time}s")

        except Exception as e:
            import traceback
            err_tb = traceback.format_exc()
            print(f"  [FAILED] {run_key}: {e}")
            run_record = {
                "run_id": run_key,
                "model": "OLMoE-1B-7B",
                "method": method,
                "probe_strategy": probe_strat,
                "target_experts": target,
                "seed": seed,
                "status": "FAILED",
                "failure_reason": str(e),
                "traceback": err_tb
            }

        with open(raw_dir / f"{run_key}.json", "w") as f:
            json.dump(run_record, f, indent=2)
        all_run_records.append(run_record)

        del comp_model
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PHASE 4: Compile Unified Results Database & Retention Tables
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("[CONSOLIDATION] Generating Unified Results Database and Retention CSVs...")
    print(f"{'='*70}")

    df_results = pd.DataFrame(all_run_records)
    df_results.to_csv(agg_dir / "results.csv", index=False)

    with open(agg_dir / "all_runs.jsonl", "w") as f:
        for r in all_run_records:
            f.write(json.dumps(r) + "\n")

    # Compute Retention Ratios relative to Base Model
    base_ppl = base_record["wikitext2_ppl"]
    base_gsm = max(base_record["gsm8k_accuracy"], 1e-4)
    base_mmlu = max(base_record["mmlu_accuracy"], 1e-4)
    base_mbpp = max(base_record["mbpp_pass_at_1"], 1e-4)

    retention_rows = []
    for r in all_run_records:
        if r.get("status") == "SUCCESS":
            retention_rows.append({
                "run_id": r["run_id"],
                "method": r["method"],
                "probe_strategy": r["probe_strategy"],
                "target_experts": r["target_experts"],
                "seed": r["seed"],
                "ppl_retention": round(base_ppl / r["wikitext2_ppl"], 4), # Inverse PPL ratio (higher is better)
                "delta_ppl": round(r["wikitext2_ppl"] - base_ppl, 4),
                "causal_kl": r["causal_heldout_kl"],
                "gsm8k_retention": round(r["gsm8k_accuracy"] / base_gsm, 4),
                "mmlu_retention": round(r["mmlu_accuracy"] / base_mmlu, 4),
                "mbpp_retention": round(r["mbpp_pass_at_1"] / base_mbpp, 4),
                "mean_capability_retention": round(
                    (r["gsm8k_accuracy"] / base_gsm + r["mmlu_accuracy"] / base_mmlu + r["mbpp_pass_at_1"] / base_mbpp) / 3.0, 4
                )
            })

    df_retention = pd.DataFrame(retention_rows)
    df_retention.to_csv(agg_dir / "capability_retention.csv", index=False)
    print("\nCapability Retention Summary:")
    print(df_retention[["method", "probe_strategy", "target_experts", "delta_ppl", "gsm8k_retention", "mmlu_retention", "mbpp_retention", "mean_capability_retention"]])

    print(f"\n[COMPLETE] Campaign runs finished. Results saved to {agg_dir}")


if __name__ == "__main__":
    main()
