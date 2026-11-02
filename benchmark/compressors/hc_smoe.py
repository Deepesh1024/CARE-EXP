"""
benchmark/compressors/hc_smoe.py
================================
Concrete compressor for HC-SMoE (PMLR / ICML 2025).
Reference: Chen et al., "Hierarchical Clustering for Mixture-of-Experts Compression"

Algorithm:
1. Expert Pairwise Similarity: Cosine distance of expert output activation profiles
   (or router weight proxy if activations not provided).
2. Agglomerative Hierarchical Clustering: Successive linkage reduction to target_k clusters.
3. Frequency-Weighted Parameter Merging:
   W_merged = sum(f_i * W_i) / (sum(f_i) + eps)
4. Physical Tensor Shrinking: Consolidates parameters to target_k experts and sums router rows.
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

from faithful_baselines.hc_smoe_engine import HCSMoEEngine


class HCSMoECompressor(BaseMoECompressor):
    METHOD_ID: str = "hc-smoe"
    IMPLEMENTATION_VERSION: str = "1.0.0"
    DEFAULT_FIDELITY_STATUS: str = "PARTIAL (router-proxy)"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        super().__init__(config=config, model_adapter=model_adapter)
        self.linkage = self.config.get("linkage", "average")
        self.similarity_metric = self.config.get("similarity_metric", "router_cosine")

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
        warnings = []

        # Determine representations and frequencies
        reps_per_layer = None
        freqs_per_layer = {}

        if isinstance(calibration_data, dict):
            reps_per_layer = calibration_data.get("expert_representations", None)
            freqs_per_layer = calibration_data.get("frequencies", {})

        if reps_per_layer is None:
            warnings.append(
                "FIDELITY WARNING: Pairwise dissimilarity is computed from router weight cosine similarity "
                "as a proxy. The published paper (Chen et al., ICML 2025) specifies expert output activation "
                "profiles on calibration tokens."
            )

        # Ensure frequencies per layer exist
        for layer_idx, block in blocks:
            if layer_idx not in freqs_per_layer:
                num_e = self.model_adapter.get_num_experts(block)
                freqs_per_layer[layer_idx] = torch.ones(num_e, dtype=torch.float32, device=device) / num_e

        engine = HCSMoEEngine(model, linkage=self.linkage, similarity_metric=self.similarity_metric)
        labels_map = engine.compress_model(
            target_experts=target,
            frequencies_per_layer=freqs_per_layer,
            representations_per_layer=reps_per_layer
        )

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
            calibration_metadata={"linkage": self.linkage, "metric": self.similarity_metric},
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=warnings,
        )
