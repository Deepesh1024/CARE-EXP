"""
benchmark/compressors/sub_moe.py
================================
Concrete compressor for Sub-MoE (AAAI 2026).
Reference: Li et al., "Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging"

Algorithm:
1. Adaptive Expert Clustering: K-means on expert output representations.
2. Experts Union Decomposition: Joint SVD on concatenated expert matrices (vertical for down_proj,
   horizontal for gate_up_proj) extracting shared left/right singular subspaces.
3. Frequency-Based Subspace Merging: Consolidates decomposed singular components weighted by
   expert routing frequencies.
4. Physical Tensor Shrinking: Excises redundant expert slots and sums member router rows.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

import os
import sys

from benchmark.compressors.base import BaseMoECompressor, BaseMoEModelAdapter, CompressionResult

_FAITHFUL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OLMoE-1B-7B", "methods"))
if _FAITHFUL_DIR not in sys.path:
    sys.path.insert(0, _FAITHFUL_DIR)

from faithful_baselines.sub_moe_engine import SubMoEEngine


class SubMoECompressor(BaseMoECompressor):
    METHOD_ID: str = "sub-moe"
    IMPLEMENTATION_VERSION: str = "1.0.0"
    DEFAULT_FIDELITY_STATUS: str = "PARTIAL (router-weight clustering proxy)"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        super().__init__(config=config, model_adapter=model_adapter)
        self.rank_ratio = self.config.get("rank_ratio", 0.5)

    def _collect_calibration_stats(
        self,
        model: nn.Module,
        blocks: List[Tuple[int, nn.Module]],
        calibration_data: Any,
        device: torch.device
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, np.ndarray]]:
        """
        Collects mean expert output representations and routing frequencies per layer.
        """
        layer_outputs: Dict[int, torch.Tensor] = {}
        layer_frequencies: Dict[int, np.ndarray] = {}
        
        # If calibration data is dictionary containing precomputed stats, use it
        if isinstance(calibration_data, dict) and "expert_outputs" in calibration_data:
            return calibration_data["expert_outputs"], calibration_data.get("frequencies", {})

        # Default initialization based on router weights
        for layer_idx, block in blocks:
            router = self.model_adapter.get_router(block)
            num_e = self.model_adapter.get_num_experts(block)
            # Stand-in representation from router projection if forward extraction not provided
            layer_outputs[layer_idx] = router.weight.data.clone().float()
            layer_frequencies[layer_idx] = np.ones(num_e, dtype=np.float32) / num_e

        # If tokenized calibration batches provided, run forward pass to accumulate statistics
        if isinstance(calibration_data, list) and len(calibration_data) > 0 and isinstance(calibration_data[0], dict):
            with torch.inference_mode():
                freq_counts = {l: np.zeros(self.model_adapter.get_num_experts(b), dtype=np.float64) for l, b in blocks}
                accum_outputs = {l: torch.zeros_like(layer_outputs[l]) for l in layer_outputs}
                total_tokens = 0
                for batch in calibration_data[:self.config.get("max_cal_batches", 4)]:
                    input_ids = batch["input_ids"].to(device)
                    if input_ids.ndim == 1:
                        input_ids = input_ids.unsqueeze(0)
                    total_tokens += input_ids.numel()
                    out = model(input_ids=input_ids, output_hidden_states=True)
                    hidden = getattr(out, "hidden_states", None)
                    for layer_idx, block in blocks:
                        router = self.model_adapter.get_router(block)
                        h_in = hidden[layer_idx] if hidden is not None else torch.randn(input_ids.shape[0], input_ids.shape[1], router.weight.shape[1], device=device)
                        logits = router(h_in)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        top_k = self.model_adapter.get_top_k(block)
                        _, top_indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                        for exp_id in range(freq_counts[layer_idx].shape[0]):
                            freq_counts[layer_idx][exp_id] += (top_indices == exp_id).sum().item()
                if total_tokens > 0:
                    for l in layer_frequencies:
                        s = freq_counts[l].sum()
                        if s > 0:
                            layer_frequencies[l] = (freq_counts[l] / s).astype(np.float32)

        return layer_outputs, layer_frequencies

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

        torch.manual_seed(seed)
        np.random.seed(seed)

        warnings = [
            "Sub-MoE clustering uses router weights as a proxy for expert output representations; "
            "activation output representation accumulation is not implemented."
        ]

        engine = SubMoEEngine(model, rank_ratio=self.rank_ratio)
        blocks = self.model_adapter.get_moe_blocks(model)
        device = next(model.parameters()).device

        exp_outputs, exp_freqs = self._collect_calibration_stats(model, blocks, calibration_data, device)
        per_layer_details = {}

        try:
            for layer_idx, block in blocks:
                reps = exp_outputs[layer_idx]
                freqs = exp_freqs[layer_idx]
                cluster_labels = engine.cluster_experts_by_output(block, reps, target)
                engine.compress_block(block, cluster_labels=cluster_labels, frequencies=freqs)
                per_layer_details[layer_idx] = {
                    "experts_before": current_experts,
                    "experts_after": target,
                    "cluster_labels": cluster_labels.tolist() if hasattr(cluster_labels, "tolist") else list(cluster_labels),
                }
        finally:
            self.model_adapter.sync_config(model, target)

        runtime = time.time() - t_start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        return CompressionResult(
            model=model,
            method_id=self.METHOD_ID,
            implementation_version=self.IMPLEMENTATION_VERSION,
            fidelity_status=self.DEFAULT_FIDELITY_STATUS,
            experts_before=current_experts,
            experts_after=target,
            per_layer_details=per_layer_details,
            router_metadata={"aggregation": "sum"},
            calibration_metadata={"rank_ratio": self.rank_ratio},
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=warnings,
        )
