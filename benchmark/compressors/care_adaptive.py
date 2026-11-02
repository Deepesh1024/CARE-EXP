"""
benchmark/compressors/care_adaptive.py
======================================
Audited, paper-faithful compressor for CARE-Adaptive (v2.3).

Algorithm:
1. Computes dynamic capability state C_t across all currently active experts at each step
   using v2.3 dual-architecture dispatch (fused OlmoeExperts / unfused ModuleList)
   and corrected layer normalization.
2. Nominates top-K nearest active pairs across the entire active pool [0, N_t - 1].
3. Evaluates candidate pairs transactionally via marginal KL divergence damage on eval_batches.
4. Permanently commits argmin KL functional damage pair.
5. Maintains active_to_original identity tracking and cluster size tracking across every step.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from benchmark.compressors.base import BaseMoECompressor, BaseMoEModelAdapter, CompressionResult
from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v21.care_capability_v23 import (
    compute_global_capability_vectors_v23,
    pairwise_capability_distances_v23,
    generate_candidate_pool_v23,
)
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage


class CAREAdaptiveCompressor(BaseMoECompressor):
    METHOD_ID: str = "care-adaptive"
    IMPLEMENTATION_VERSION: str = "2.3.0"
    DEFAULT_FIDELITY_STATUS: str = "FAITHFUL"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        super().__init__(config=config, model_adapter=model_adapter)
        self.candidate_pool_size = self.config.get("candidate_pool_size", 5)
        self.router_merge_mode = self.config.get("router_merge_mode", "uniform")
        if self.router_merge_mode not in ["uniform", "frequency_weighted", "sum"]:
            raise ValueError(
                f"Unsupported router_merge_mode: '{self.router_merge_mode}'. "
                f"Must be one of: ['uniform', 'frequency_weighted', 'sum']."
            )
        self.eval_batch_size = self.config.get("eval_batch_size", 4)
        self.max_eval_batches = self.config.get("max_eval_batches", 4)
        self.num_axes = self.config.get("num_axes", 10)
        self.samples_per_axis = self.config.get("samples_per_axis", 20)

    def compress(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        target_experts: Optional[int] = None,
        seed: int = 42,
    ) -> CompressionResult:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.time()
        current_experts, target = self.validate_inputs(model, target_experts)

        if current_experts == target:
            return CompressionResult(
                model=model,
                method_id=self.METHOD_ID,
                implementation_version=self.IMPLEMENTATION_VERSION,
                fidelity_status=self.DEFAULT_FIDELITY_STATUS,
                experts_before=current_experts,
                experts_after=current_experts,
                runtime_s=0.0,
            )

        # Fail-fast check: calibration_data must be provided as DataFrame
        if calibration_data is None:
            raise ValueError(
                "CAREAdaptiveCompressor requires calibration_data (probe DataFrame with 'axis_idx', "
                "'input_ids', 'attention_mask'). Silent fallback to router weights is strictly prohibited."
            )

        # Fail-fast check: eval_batches must be provided for functional KL candidate evaluation
        eval_batches = self.config.get("eval_batches", None)
        if eval_batches is None and isinstance(calibration_data, list):
            eval_batches = calibration_data[:self.max_eval_batches]
        if eval_batches is None or len(eval_batches) == 0:
            raise ValueError(
                "CAREAdaptiveCompressor requires eval_batches for marginal KL candidate evaluation. "
                "Geometry-only fallback without functional evaluation is strictly prohibited."
            )

        torch.manual_seed(seed)
        engine = PhysicalMergeEngine(
            model,
            router_merge_mode=self.router_merge_mode,
            expert_frequencies=self.config.get("expert_frequencies", None)
        )

        device = next(model.parameters()).device
        step_records = []
        step = 1

        try:
            while engine.current_num_experts > target:
                experts_before_step = engine.current_num_experts
                t_step_start = time.time()

                # 1. Compute dynamic capability state C_t across active experts using v2.3
                C = compute_global_capability_vectors_v23(
                    model=model,
                    moe_blocks=engine.moe_blocks,
                    current_num_experts=engine.current_num_experts,
                    df_tokens=calibration_data,
                    samples_per_axis=self.samples_per_axis,
                    device=str(device),
                )

                # Fail-fast validation of capability vectors
                assert np.all(np.isfinite(C)), "Capability matrix contains non-finite values (NaN / Inf)."
                assert np.min(np.std(C, axis=0)) > 1e-6, "Capability matrix contains zero or degenerate variance columns."

                # 2. Compute pairwise distances and candidate pool
                distances = pairwise_capability_distances_v23(C)
                k_pool = min(self.candidate_pool_size, (engine.current_num_experts * (engine.current_num_experts - 1)) // 2)
                candidates = generate_candidate_pool_v23(distances, pool_size=k_pool)

                # Fail-fast check: verify candidate pairs are strictly valid and within active expert bounds
                assert len(candidates) > 0, "Nominated candidate pool is empty."
                assert all(0 <= i < j < engine.current_num_experts for i, j in candidates), (
                    f"Candidate pairs out of valid active range [0, {engine.current_num_experts}): {candidates}"
                )

                # 3. Transactionally evaluate candidates via marginal KL
                current_logprobs = collect_current_logits(
                    model, eval_batches, self.eval_batch_size, self.max_eval_batches, device
                )

                candidate_results = []
                for i, j in candidates:
                    orig_i = int(engine.get_original_id(i))
                    orig_j = int(engine.get_original_id(j))
                    cap_dist = float(distances[i][j])

                    engine.snapshot(i, j)
                    try:
                        engine.merge_experts(i, j)
                        damage = evaluate_marginal_damage(
                            model, eval_batches, current_logprobs,
                            self.eval_batch_size, self.max_eval_batches, device
                        )
                        assert not np.isnan(damage) and not np.isinf(damage), "Evaluated marginal KL is NaN/Inf."
                    finally:
                        engine.restore()

                    candidate_results.append({
                        "local_pair": [int(i), int(j)],
                        "original_pair": [orig_i, orig_j],
                        "capability_distance": cap_dist,
                        "marginal_damage": float(damage),
                    })

                best_candidate = min(candidate_results, key=lambda x: x["marginal_damage"])
                commit_i, commit_j = best_candidate["local_pair"]
                orig_i = int(engine.get_original_id(commit_i))
                orig_j = int(engine.get_original_id(commit_j))

                # 4. Commit argmin merge permanently
                engine.merge_experts(commit_i, commit_j)

                # Mapping integrity check
                active_mapping = engine.get_active_mapping()
                assert len(active_mapping) == engine.current_num_experts, "Active mapping length mismatch."

                cluster_sizes = {orig: len(engine.get_cluster(loc)) for loc, orig in active_mapping.items()}

                step_records.append({
                    "step": step,
                    "experts_before": experts_before_step,
                    "experts_after": engine.current_num_experts,
                    "selected_local_pair": [commit_i, commit_j],
                    "selected_original_pair": [orig_i, orig_j],
                    "capability_distance": best_candidate["capability_distance"],
                    "marginal_damage": best_candidate["marginal_damage"],
                    "candidates_evaluated": len(candidate_results),
                    "cluster_sizes": cluster_sizes,
                    "step_time_s": round(time.time() - t_step_start, 3),
                })
                step += 1
        finally:
            self.model_adapter.sync_config(model, engine.current_num_experts)

        runtime = time.time() - t_start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        per_layer_details = {
            layer_idx: {
                "experts_before": current_experts,
                "experts_after": engine.current_num_experts,
                "active_to_original": dict(engine.get_active_mapping()),
            }
            for layer_idx in range(len(engine.moe_blocks))
        }

        return CompressionResult(
            model=model,
            method_id=self.METHOD_ID,
            implementation_version=self.IMPLEMENTATION_VERSION,
            fidelity_status=self.DEFAULT_FIDELITY_STATUS,
            experts_before=current_experts,
            experts_after=engine.current_num_experts,
            per_layer_details=per_layer_details,
            trajectory=step_records,
            active_expert_mapping=engine.get_active_mapping(),
            router_metadata={"mode": self.router_merge_mode},
            calibration_metadata={
                "samples_per_axis": self.samples_per_axis,
                "candidate_pool_size": self.candidate_pool_size,
                "eval_batches_count": len(eval_batches),
            },
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=[],
        )
