"""
benchmark/compressors/base.py
==============================
Common OOP interface for MoE model compression algorithms.

Provides:
- BaseMoEModelAdapter: Model-architecture abstraction for expert & router access.
- OLMoEModelAdapter: Concrete adapter for OLMoE-style fused MoE architectures.
- CompressionResult: Standardized result object across all compression algorithms.
- BaseMoECompressor: Abstract base class for all compression algorithms.
"""

from __future__ import annotations
import abc
import time
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Model Adapter Abstraction
# ==============================================================================

class BaseMoEModelAdapter(abc.ABC):
    """
    Abstract interface isolating model-specific MoE tensor layouts,
    expert structures, and router implementations.
    """

    @abc.abstractmethod
    def get_moe_blocks(self, model: nn.Module) -> List[Tuple[int, nn.Module]]:
        """Returns list of (layer_index, moe_block_module) for all MoE layers."""
        pass

    @abc.abstractmethod
    def get_router(self, block: nn.Module) -> nn.Module:
        """Returns the router / gate module for a given MoE block."""
        pass

    @abc.abstractmethod
    def get_num_experts(self, block: nn.Module) -> int:
        """Returns the current number of routed experts in the block."""
        pass

    @abc.abstractmethod
    def get_top_k(self, block: nn.Module, model: Optional[nn.Module] = None) -> int:
        """Returns the number of active experts selected per token."""
        pass

    @abc.abstractmethod
    def sync_config(self, model: nn.Module, target_experts: int) -> None:
        """Synchronizes model.config fields (e.g. num_experts, num_local_experts)."""
        pass


class OLMoEModelAdapter(BaseMoEModelAdapter):
    """
    Concrete adapter for OLMoE-1B-7B (and compatible architectures like MockMoEBlock).
    
    Handles:
    - Fused expert weights: gate_up_proj [num_experts, 2 * d_ff, d_model]
    - Down projection: down_proj [num_experts, d_model, d_ff]
    - Router weight: gate.weight [num_experts, d_model]
    - Config sync: config.num_experts, config.num_local_experts
    """

    def get_moe_blocks(self, model: nn.Module) -> List[Tuple[int, nn.Module]]:
        blocks = []
        for name, module in model.named_modules():
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                blocks.append((len(blocks), module))
        return blocks

    def get_router(self, block: nn.Module) -> nn.Module:
        if hasattr(block, "gate"):
            return block.gate
        if hasattr(block, "router"):
            return block.router
        raise AttributeError("MoE block has neither 'gate' nor 'router' attribute.")

    def get_num_experts(self, block: nn.Module) -> int:
        router = self.get_router(block)
        if hasattr(router, "weight"):
            return router.weight.shape[0]
        if hasattr(block.experts, "num_experts"):
            return block.experts.num_experts
        raise AttributeError("Could not determine expert count from block.")

    def get_top_k(self, block: nn.Module, model: Optional[nn.Module] = None) -> int:
        if hasattr(block, "num_experts_per_tok"):
            return block.num_experts_per_tok
        router = self.get_router(block)
        if hasattr(router, "top_k"):
            return router.top_k
        if model is not None and hasattr(model, "config") and hasattr(model.config, "num_experts_per_tok"):
            return model.config.num_experts_per_tok
        return 8

    def sync_config(self, model: nn.Module, target_experts: int) -> None:
        if hasattr(model, "config"):
            if hasattr(model.config, "num_experts"):
                model.config.num_experts = target_experts
            if hasattr(model.config, "num_local_experts"):
                model.config.num_local_experts = target_experts


# ==============================================================================
# Standardized Compression Result
# ==============================================================================

@dataclass
class CompressionResult:
    """
    Standardized result object returned by all compression algorithms.
    """
    model: nn.Module
    method_id: str
    implementation_version: str
    fidelity_status: str  # FAITHFUL | PARTIAL (reason) | INCORRECT | CONTROL
    experts_before: int
    experts_after: int
    per_layer_details: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    active_to_original: Optional[Dict[int, int]] = None
    router_metadata: Dict[str, Any] = field(default_factory=dict)
    calibration_metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0
    peak_memory_mb: float = 0.0
    warnings: List[str] = field(default_factory=list)
    step_records: Optional[List[Dict[str, Any]]] = None
    trajectory: Optional[List[Dict[str, Any]]] = None
    active_expert_mapping: Optional[Dict[int, int]] = None

    def __post_init__(self):
        if self.trajectory is None and self.step_records is not None:
            self.trajectory = self.step_records
        elif self.step_records is None and self.trajectory is not None:
            self.step_records = self.trajectory

        if self.active_expert_mapping is None and self.active_to_original is not None:
            self.active_expert_mapping = self.active_to_original
        elif self.active_to_original is None and self.active_expert_mapping is not None:
            self.active_to_original = self.active_expert_mapping

    def is_faithful(self) -> bool:
        return self.fidelity_status.startswith("FAITHFUL")


# ==============================================================================
# Base Compressor Abstract Base Class
# ==============================================================================

class BaseMoECompressor(abc.ABC):
    """
    Common base class for all MoE compression algorithms.
    
    Usage pattern:
        compressor = SpecificCompressor(config, model_adapter=adapter)
        result = compressor.compress(
            model=model,
            calibration_data=calibration_data,
            target_experts=target_experts,
            seed=seed,
        )
    """

    METHOD_ID: str = "base_moe_compressor"
    IMPLEMENTATION_VERSION: str = "1.0.0"
    DEFAULT_FIDELITY_STATUS: str = "UNVERIFIED"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[BaseMoEModelAdapter] = None
    ):
        self.config = config or {}
        self.model_adapter = model_adapter or OLMoEModelAdapter()
        self.fidelity_status = self.DEFAULT_FIDELITY_STATUS

    @abc.abstractmethod
    def compress(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        target_experts: Optional[int] = None,
        seed: int = 42,
    ) -> CompressionResult:
        """
        Compress the model from current expert count to target_experts.
        
        Args:
            model: PyTorch MoE model to compress.
            calibration_data: Optional calibration batches or token representations.
            target_experts: Target expert count per layer.
            seed: Random seed for reproducibility.
            
        Returns:
            CompressionResult containing the compressed model and complete metadata.
        """
        pass

    def validate_inputs(
        self,
        model: nn.Module,
        target_experts: Optional[int],
    ) -> Tuple[int, int]:
        """
        Validates basic input invariants:
        - MoE blocks exist
        - target_experts is valid: 1 <= target_experts <= current_experts
        Returns (current_experts, target_experts).
        """
        blocks = self.model_adapter.get_moe_blocks(model)
        if not blocks:
            raise ValueError(f"[{self.METHOD_ID}] No MoE blocks found in model using adapter {type(self.model_adapter).__name__}.")
            
        current_experts = self.model_adapter.get_num_experts(blocks[0][1])
        if target_experts is None:
            target_experts = current_experts - 1
            
        if not isinstance(target_experts, int):
            raise TypeError(f"[{self.METHOD_ID}] target_experts must be an integer, got {type(target_experts)}.")
            
        if target_experts < 1:
            raise ValueError(f"[{self.METHOD_ID}] target_experts must be >= 1, got {target_experts}.")
            
        if target_experts > current_experts:
            raise ValueError(f"[{self.METHOD_ID}] target_experts ({target_experts}) cannot exceed current experts ({current_experts}).")
            
        return current_experts, target_experts
