"""
benchmark/compressors/factory.py
================================
Factory registry for instantiating MoE compressors by method identifier.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Type
import torch.nn as nn

from benchmark.compressors.base import BaseMoECompressor, BaseMoEModelAdapter, OLMoEModelAdapter
from benchmark.compressors.care_static import CAREStaticCompressor, CAREStaticK1Compressor
from benchmark.compressors.care_adaptive import CAREAdaptiveCompressor
from benchmark.compressors.sub_moe import SubMoECompressor
from benchmark.compressors.reap_prune import REAPPruneCompressor
from benchmark.compressors.hc_smoe import HCSMoECompressor
from benchmark.compressors.mc_smoe import MCSMoECompressor
from benchmark.compressors.random_control import RandomControlCompressor


COMPRESSOR_REGISTRY: Dict[str, Type[BaseMoECompressor]] = {
    "care_static": CAREStaticCompressor,
    "care-static": CAREStaticCompressor,
    "care_static_k5": CAREStaticCompressor,
    "care-static-k5": CAREStaticCompressor,
    "care_static_k1": CAREStaticK1Compressor,
    "care-static-k1": CAREStaticK1Compressor,
    "care_adaptive": CAREAdaptiveCompressor,
    "care-adaptive": CAREAdaptiveCompressor,
    "care_adaptive_k5": CAREAdaptiveCompressor,
    "care-adaptive-k5": CAREAdaptiveCompressor,
    "sub_moe": SubMoECompressor,
    "sub-moe": SubMoECompressor,
    "reap_prune": REAPPruneCompressor,
    "reap-prune": REAPPruneCompressor,
    "reap": REAPPruneCompressor,
    "hc_smoe": HCSMoECompressor,
    "hc-smoe": HCSMoECompressor,
    "mc_smoe": MCSMoECompressor,
    "mc-smoe": MCSMoECompressor,
    "m_smoe": MCSMoECompressor,
    "random": RandomControlCompressor,
    "random_control": RandomControlCompressor,
    "random-control": RandomControlCompressor,
}


def get_compressor(
    method_name: str,
    config: Optional[Dict[str, Any]] = None,
    model_adapter: Optional[BaseMoEModelAdapter] = None,
) -> BaseMoECompressor:
    """
    Instantiates the requested compression algorithm.

    Args:
        method_name: Algorithm key (e.g. 'care-static', 'care-adaptive', 'sub-moe', etc.)
        config: Optional algorithm configuration dict.
        model_adapter: Model adapter instance (defaults to OLMoEModelAdapter).

    Returns:
        Instantiated BaseMoECompressor.
    """
    key = method_name.strip().lower()
    if key not in COMPRESSOR_REGISTRY:
        raise ValueError(
            f"Unknown compression method '{method_name}'. Available methods: {sorted(list(set(COMPRESSOR_REGISTRY.keys())))}"
        )
    compressor_cls = COMPRESSOR_REGISTRY[key]
    return compressor_cls(config=config, model_adapter=model_adapter)
