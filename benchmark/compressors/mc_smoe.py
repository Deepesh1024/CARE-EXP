"""
benchmark/compressors/mc_smoe.py
================================
Concrete compressor for MC-SMoE (OpenReview / ICLR 2024).
Reference: Li et al., "MC-SMoE: Merging and Compressing Mixture-of-Experts"

Algorithm (Stages 1–4):
1. Routing-Policy Clustering: Groups experts based on router weight cosine similarity.
2. Dominant Expert Identification: Highest frequency expert in cluster serves as reference frame.
3. Hungarian Bipartite Neuron Matching: Solves optimal intermediate neuron permutation:
   Cost_{u, v} = <W_gate_r[u], W_gate_j[v]> + <W_up_r[u], W_up_j[v]> + <W_down_r[:, u], W_down_j[:, v]>
4. Permutation-Aligned Weight Merging: Permutes member intermediate channels into reference frame,
   followed by frequency-weighted averaging:
   W_merged = sum(f_i * W_i^(pi)) / (sum(f_i) + eps)
5. Physical Tensor Shrinking: Consolidates parameters to target_k experts and sums router rows.

Note:
Stage 5 (Low-rank SVD decomposition and structural sparsity pruning of residuals) is NOT implemented.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

import os
import sys

from benchmark.compressors.base import BaseMoECompressor, BaseMoEModelAdapter, CompressionResult

_FAITHFUL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OLMoE-1B-7B", "methods"))
if _FAITHFUL_DIR not in sys.path:
    sys.path.insert(0, _FAITHFUL_DIR)

from faithful_baselines.mc_smoe_engine import MCSMoEEngine


class MCSMoECompressor(BaseMoECompressor):
    METHOD_ID: str = "mc-smoe"
    IMPLEMENTATION_VERSION: str = "1.0.0"
    DEFAULT_FIDELITY_STATUS: str = "PARTIAL (Stages 1–4 only)"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        super().__init__(config=config, model_adapter=model_adapter)

    def compress(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        target_experts: Optional[int] = None,
        seed: int = 42,
    ) -> CompressionResult:
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
        blocks = self.model_adapter.get_moe_blocks(model)
        device = next(model.parameters()).device
        warnings = [
            "FIDELITY WARNING: Stage 5 (Low-rank SVD decomposition and structural sparsity pruning of residuals) "
            "is NOT implemented. Method executes Stages 1-4 only (routing policy grouping, dominant expert identification, "
            "Hungarian bipartite neuron matching, and permutation-aligned frequency-weighted merging)."
        ]

        freqs_per_layer = {}
        if isinstance(calibration_data, dict):
            freqs_per_layer = calibration_data.get("frequencies", {})

        for layer_idx, block in blocks:
            if layer_idx not in freqs_per_layer:
                num_e = self.model_adapter.get_num_experts(block)
                freqs_per_layer[layer_idx] = torch.ones(num_e, dtype=torch.float32, device=device) / num_e

        engine = MCSMoEEngine(model)
        labels_map = engine.compress_model(target_experts=target, frequencies_per_layer=freqs_per_layer)

        self.model_adapter.sync_config(model, target)

        runtime = time.time() - t_start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        per_layer_details = {}
        for layer_idx, labels in labels_map.items():
            per_layer_details[layer_idx] = {
                "experts_before": current_experts,
                "experts_after": target,
                "cluster_labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
            }

        return CompressionResult(
            model=model,
            method_id=self.METHOD_ID,
            implementation_version=self.IMPLEMENTATION_VERSION,
            fidelity_status=self.DEFAULT_FIDELITY_STATUS,
            experts_before=current_experts,
            experts_after=target,
            per_layer_details=per_layer_details,
            router_metadata={"aggregation": "sum"},
            calibration_metadata={"hungarian_matching": "bipartite_linear_sum_assignment"},
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=warnings,
        )
