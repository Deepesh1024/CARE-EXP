"""
benchmark/compressors/care_static.py
====================================
Audited, paper-faithful compressor for CARE-Static (v2.3).

Algorithm:
1. Computes initial capability geometry C_0 ONCE at t=0 across all initial experts using v2.3.
2. Freezes pairwise capability distance ranking across all pairs.
3. In each step:
   - For Static-K5 (default, matching manuscript Section 6.2): takes the top K=5 valid candidate
     pairs from the frozen ranking, transactionally evaluates each via marginal KL on eval_batches,
     and commits the argmin KL.
   - For Static-K1 (blind): directly commits the single nearest valid pair without trial evaluation.
4. Maintains active_to_original identity tracking, cluster sizes, and router weight combination.
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
)
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage


class CAREStaticCompressor(BaseMoECompressor):
    METHOD_ID: str = "care-static"
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

    def _get_frozen_ranking(self, C: np.ndarray) -> List[Tuple[int, int, float]]:
        distances = pairwise_capability_distances_v23(C)
        pairs = []
        num_experts = C.shape[0]
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                if j in distances.get(i, {}):
                    pairs.append((i, j, float(distances[i][j])))
        pairs.sort(key=lambda x: x[2])
        return pairs

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
                "CAREStaticCompressor requires calibration_data (probe DataFrame with 'axis_idx', "
                "'input_ids', 'attention_mask'). Silent fallback to router weights is strictly prohibited."
            )

        eval_batches = self.config.get("eval_batches", None)
        if eval_batches is None and isinstance(calibration_data, list):
            eval_batches = calibration_data[:self.max_eval_batches]

        # For Static-K > 1, eval_batches are required to evaluate candidate damage
        if self.candidate_pool_size > 1 and (eval_batches is None or len(eval_batches) == 0):
            raise ValueError(
                f"CAREStaticCompressor configured with candidate_pool_size={self.candidate_pool_size} (Static-K5) "
                f"requires eval_batches for marginal KL evaluation. Geometry-only fallback is strictly prohibited."
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

        # 1. Compute initial capability geometry C_0 once at t=0
        C_0 = compute_global_capability_vectors_v23(
            model=model,
            moe_blocks=engine.moe_blocks,
            current_num_experts=current_experts,
            df_tokens=calibration_data,
            samples_per_axis=self.samples_per_axis,
            device=str(device),
        )

        assert np.all(np.isfinite(C_0)), "Initial capability matrix contains non-finite values."
        assert np.min(np.std(C_0, axis=0)) > 1e-6, "Initial capability matrix has zero or degenerate variance."

        frozen_ranking = self._get_frozen_ranking(C_0)
        removed_original_ids = set()

        try:
            while engine.current_num_experts > target:
                experts_before_step = engine.current_num_experts
                t_step_start = time.time()

                # 2. Extract top-K valid candidate pairs from frozen ranking
                # A pair (orig_i, orig_j) is valid if neither original ID has been excised
                valid_candidates: List[Tuple[int, int, int, int, float]] = []
                for orig_i, orig_j, dist in frozen_ranking:
                    if orig_i not in removed_original_ids and orig_j not in removed_original_ids:
                        # Map original IDs to current local active indices
                        active_map = engine.get_active_mapping()
                        rev_map = {orig: local for local, orig in active_map.items()}
                        if orig_i in rev_map and orig_j in rev_map:
                            loc_i = rev_map[orig_i]
                            loc_j = rev_map[orig_j]
                            if loc_i != loc_j:
                                valid_candidates.append((min(loc_i, loc_j), max(loc_i, loc_j), orig_i, orig_j, dist))
                                if len(valid_candidates) >= self.candidate_pool_size:
                                    break

                if len(valid_candidates) == 0:
                    raise RuntimeError(f"Frozen ranking exhausted at step {step} before reaching target {target} experts.")

                # 3. Candidate evaluation and selection
                if self.candidate_pool_size > 1 and eval_batches is not None:
                    # Static-K5: evaluate candidate pairs transactionally via marginal KL
                    current_logprobs = collect_current_logits(
                        model, eval_batches, self.eval_batch_size, self.max_eval_batches, device
                    )
                    candidate_results = []
                    for loc_i, loc_j, orig_i, orig_j, dist in valid_candidates:
                        engine.snapshot(loc_i, loc_j)
                        try:
                            engine.merge_experts(loc_i, loc_j)
                            damage = evaluate_marginal_damage(
                                model, eval_batches, current_logprobs,
                                self.eval_batch_size, self.max_eval_batches, device
                            )
                            assert not np.isnan(damage) and not np.isinf(damage)
                        finally:
                            engine.restore()

                        candidate_results.append({
                            "local_pair": [loc_i, loc_j],
                            "original_pair": [orig_i, orig_j],
                            "capability_distance": dist,
                            "marginal_damage": float(damage),
                        })

                    best = min(candidate_results, key=lambda x: x["marginal_damage"])
                    commit_i, commit_j = best["local_pair"]
                    commit_orig_i, commit_orig_j = best["original_pair"]
                    chosen_dist = best["capability_distance"]
                    chosen_damage = best["marginal_damage"]
                    candidates_evaled = len(candidate_results)
                else:
                    # Static-K1 (blind): directly pick top 1 valid pair from frozen ranking
                    loc_i, loc_j, orig_i, orig_j, dist = valid_candidates[0]
                    commit_i, commit_j = loc_i, loc_j
                    commit_orig_i, commit_orig_j = orig_i, orig_j
                    chosen_dist = dist
                    chosen_damage = 0.0
                    candidates_evaled = 1

                # Record excised original ID (the one at index commit_j)
                excised_orig = int(engine.get_original_id(commit_j))
                removed_original_ids.add(excised_orig)

                # 4. Commit merge permanently
                engine.merge_experts(commit_i, commit_j)

                # Mapping integrity check
                active_mapping = engine.get_active_mapping()
                assert len(active_mapping) == engine.current_num_experts, "Active mapping length mismatch."
                assert excised_orig not in active_mapping.values(), f"Excised expert {excised_orig} still present in active mapping."

                cluster_sizes = {orig: len(engine.get_cluster(loc)) for loc, orig in active_mapping.items()}

                step_records.append({
                    "step": step,
                    "experts_before": experts_before_step,
                    "experts_after": engine.current_num_experts,
                    "selected_local_pair": [commit_i, commit_j],
                    "selected_original_pair": [commit_orig_i, commit_orig_j],
                    "capability_distance": chosen_dist,
                    "marginal_damage": chosen_damage,
                    "candidates_evaluated": candidates_evaled,
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
                "candidate_pool_size": self.candidate_pool_size,
                "variant": "Static-K5" if self.candidate_pool_size > 1 else "Static-K1",
            },
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=[],
        )


class CAREStaticK1Compressor(CAREStaticCompressor):
    """
    CARE-Static-K1 (Blind frozen baseline):
    Commits the top-1 valid pair from the frozen capability ranking without evaluating candidates.
    Matches the historical v2.2 baseline implementation.
    """
    METHOD_ID: str = "care-static-k1"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        cfg = dict(config or {})
        cfg["candidate_pool_size"] = 1
        super().__init__(config=cfg, model_adapter=model_adapter)

