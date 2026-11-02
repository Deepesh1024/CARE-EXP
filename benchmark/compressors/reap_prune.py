"""
benchmark/compressors/reap_prune.py
===================================
Audited, paper-faithful compressor for REAP-Prune (Cerebras Research 2024).
Reference: "REAP: Router-Weighted Activation Pruning for Mixture-of-Experts"

Algorithm:
1. Saliency Observation: Accumulates router-weighted activation norms
   s_i(t) = w_i(t) * ||h_i(t)||_2 during forward passes on calibration tokens.
2. Saliency Aggregation: Averages weighted activation norms per active token
   (reap metric, matching Cerebras primary equation) or sums over tokens (weighted_ean_sum).
3. Physical Expert Excision: Identifies and physically excises lowest-saliency experts per layer.
4. Router Excision & Renormalization: Removes corresponding router rows and normalizes routing weights.
"""

from __future__ import annotations
import time
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

from benchmark.compressors.base import BaseMoECompressor, BaseMoEModelAdapter, CompressionResult

_FAITHFUL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OLMoE-1B-7B", "methods"))
if _FAITHFUL_DIR not in sys.path:
    sys.path.insert(0, _FAITHFUL_DIR)

from faithful_baselines.reap_prune_engine import (
    REAPSaliencyObserver,
    REAPPruneEngine
)


class REAPPruneCompressor(BaseMoECompressor):
    METHOD_ID: str = "reap-prune"
    IMPLEMENTATION_VERSION: str = "2.0.0"
    DEFAULT_FIDELITY_STATUS: str = "FAITHFUL (Cerebras Saliency Pruning)"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        super().__init__(config=config, model_adapter=model_adapter)
        # 'reap' matches Cerebras Eq. (average over active tokens); 'weighted_ean_sum' is sum over all tokens
        self.metric = self.config.get("metric", "reap")
        self.renormalize_router_weights = self.config.get("renormalize_router_weights", True)
        self.max_cal_batches = self.config.get("max_cal_batches", 16)

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
        device = next(model.parameters()).device
        total_tokens_observed = 0

        # 1. Collect saliency observations with strict fail-fast validation
        if isinstance(calibration_data, list) and len(calibration_data) > 0 and isinstance(calibration_data[0], dict):
            observer = REAPSaliencyObserver(model, renormalize_router_weights=self.renormalize_router_weights)
            observer.register_hooks()
            try:
                with torch.inference_mode():
                    cal_batches = calibration_data[:self.max_cal_batches]
                    for batch in cal_batches:
                        input_ids = batch["input_ids"].to(device)
                        if input_ids.ndim == 1:
                            input_ids = input_ids.unsqueeze(0)

                        attn_mask = batch.get("attention_mask", None)
                        if attn_mask is not None:
                            attn_mask = attn_mask.to(device)
                            if attn_mask.ndim == 1:
                                attn_mask = attn_mask.unsqueeze(0)

                        model(input_ids=input_ids, attention_mask=attn_mask)
                        if attn_mask is not None:
                            total_tokens_observed += int(attn_mask.sum().item())
                        else:
                            total_tokens_observed += input_ids.numel()
            finally:
                observer.remove_hooks()

            layer_saliencies = observer.get_saliencies(metric=self.metric)
        elif isinstance(calibration_data, dict) and "saliencies" in calibration_data:
            layer_saliencies = calibration_data["saliencies"]
        else:
            raise ValueError(
                "REAPPruneCompressor requires valid calibration batches (list of dicts with 'input_ids' "
                "and 'attention_mask') or a precomputed 'saliencies' dict. "
                "Silent fallback to router-weight norms is strictly prohibited."
            )

        # Compute saliency diagnostics
        saliency_diagnostics = {}
        for l_idx, s_vec in layer_saliencies.items():
            s_float = s_vec.float()
            saliency_diagnostics[int(l_idx)] = {
                "min": round(float(s_float.min().item()), 6),
                "max": round(float(s_float.max().item()), 6),
                "mean": round(float(s_float.mean().item()), 6),
                "zero_saliency_count": int((s_float == 0).sum().item()),
            }

        # 2. Prune model using verified REAPPruneEngine
        try:
            prune_engine = REAPPruneEngine(model, renormalize_router_weights=self.renormalize_router_weights)
            retained_map = prune_engine.prune_model(target_experts=target, layer_saliencies=layer_saliencies)
        finally:
            self.model_adapter.sync_config(model, target)

        # 3. Post-prune assertions
        blocks = self.model_adapter.get_moe_blocks(model)
        for layer_idx, block in blocks:
            n_exp = self.model_adapter.get_num_experts(block)
            assert n_exp == target, f"Layer {layer_idx} has {n_exp} experts; expected exactly {target}."
        if hasattr(model, "config") and hasattr(model.config, "num_experts"):
            assert model.config.num_experts == target, f"Model config num_experts {model.config.num_experts} != {target}."

        runtime = time.time() - t_start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        per_layer_details = {}
        for layer_idx, retained in retained_map.items():
            per_layer_details[layer_idx] = {
                "experts_before": current_experts,
                "experts_after": target,
                "retained_experts": retained,
                "pruned_experts": [e for e in range(current_experts) if e not in set(retained)],
                "diagnostics": saliency_diagnostics.get(layer_idx, {}),
            }

        return CompressionResult(
            model=model,
            method_id=self.METHOD_ID,
            implementation_version=self.IMPLEMENTATION_VERSION,
            fidelity_status=self.DEFAULT_FIDELITY_STATUS,
            experts_before=current_experts,
            experts_after=target,
            per_layer_details=per_layer_details,
            router_metadata={"operation": "excision", "renormalize": self.renormalize_router_weights},
            calibration_metadata={
                "metric": self.metric,
                "total_tokens_observed": total_tokens_observed,
                "max_cal_batches": self.max_cal_batches,
            },
            runtime_s=round(runtime, 3),
            peak_memory_mb=round(peak_mem, 2),
            warnings=[],
        )
