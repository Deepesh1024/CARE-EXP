"""
benchmark/compressors/random_control.py
=======================================
Concrete compressor for Random Merging Control Baseline.

Supports:
1. Random-K5 (Matched Candidate Protocol):
   At each step, samples K=5 random pairs from active experts, evaluates each physically
   via marginal KL on calibration batches, and commits the candidate with minimal damage.
   Matches CARE-Adaptive's evaluation budget and objective.
2. Random-1 (Unbudgeted):
   Selects 1 random pair uniformly and commits without candidate evaluation.
"""

from __future__ import annotations
import time
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from benchmark.compressors.base import BaseMoECompressor, BaseMoEModelAdapter, CompressionResult
from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage


class RandomControlCompressor(BaseMoECompressor):
    METHOD_ID: str = "random-control"
    IMPLEMENTATION_VERSION: str = "2.1.0"
    DEFAULT_FIDELITY_STATUS: str = "CONTROL (random baseline)"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        super().__init__(config=config, model_adapter=model_adapter)
        self.candidate_k = self.config.get("candidate_k", 5)
        self.router_merge_mode = self.config.get("router_merge_mode", "uniform")
        if self.router_merge_mode not in ["uniform", "frequency_weighted", "sum"]:
            raise ValueError(f"Unsupported router_merge_mode: '{self.router_merge_mode}'.")
        self.eval_batch_size = self.config.get("eval_batch_size", 4)
        self.max_eval_batches = self.config.get("max_eval_batches", 4)

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

        eval_batches = self.config.get("eval_batches", None)
        if eval_batches is None and isinstance(calibration_data, list):
            eval_batches = calibration_data[:self.max_eval_batches]

        if self.candidate_k > 1 and (eval_batches is None or len(eval_batches) == 0):
            raise ValueError(
                f"RandomControlCompressor with candidate_k={self.candidate_k} requires eval_batches "
                f"for marginal KL candidate evaluation. Silent fallback is prohibited."
            )

        random.seed(seed)
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

                # Formulate all active pairs
                all_pairs = [
                    (i, j) for i in range(experts_before_step) for j in range(i + 1, experts_before_step)
                ]
                num_cands = min(self.candidate_k, len(all_pairs))
                candidate_pool = random.sample(all_pairs, num_cands)

                candidate_results = []

                if self.candidate_k > 1 and eval_batches is not None and len(eval_batches) > 0:
                    current_logprobs = collect_current_logits(
                        model, eval_batches, self.eval_batch_size, self.max_eval_batches, device
                    )
                    for i, j in candidate_pool:
                        orig_i = int(engine.get_original_id(i))
                        orig_j = int(engine.get_original_id(j))

                        engine.snapshot(i, j)
                        try:
                            engine.merge_experts(i, j)
                            damage = evaluate_marginal_damage(
                                model, eval_batches, current_logprobs,
                                self.eval_batch_size, self.max_eval_batches, device
                            )
                            assert not np.isnan(damage) and not np.isinf(damage)
                        finally:
                            engine.restore()

                        candidate_results.append({
                            "local_pair": [int(i), int(j)],
                            "original_pair": [orig_i, orig_j],
                            "marginal_damage": float(damage),
                        })
                    best_candidate = min(candidate_results, key=lambda x: x["marginal_damage"])
                else:
                    # Random-1: select first candidate
                    first_i, first_j = candidate_pool[0]
                    orig_i = int(engine.get_original_id(first_i))
                    orig_j = int(engine.get_original_id(first_j))
                    best_candidate = {
                        "local_pair": [int(first_i), int(first_j)],
                        "original_pair": [orig_i, orig_j],
                        "marginal_damage": 0.0,
                    }
                    candidate_results.append(best_candidate)

                selected_i, selected_j = best_candidate["local_pair"]
                orig_i = int(engine.get_original_id(selected_i))
                orig_j = int(engine.get_original_id(selected_j))
                engine.merge_experts(selected_i, selected_j)

                step_time = time.time() - t_step_start
                step_records.append({
                    "step": step,
                    "experts_before": experts_before_step,
                    "experts_after": engine.current_num_experts,
                    "selected_local_pair": [selected_i, selected_j],
                    "selected_original_pair": [orig_i, orig_j],
                    "candidate_pool": candidate_results,
                    "marginal_damage": best_candidate["marginal_damage"],
                    "step_time_s": round(step_time, 3),
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
                "candidate_k": self.candidate_k,
                "variant": "Random-K5" if self.candidate_k > 1 else "Random-1",
            },
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=[],
        )
