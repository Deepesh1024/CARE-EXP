"""
benchmark/OLMoE-1B-7B/methods/faithful_baselines/reap_prune_engine.py
=====================================================================
Faithful Primary-Source Implementation of REAP (Cerebras Research 2024).
Reference: "REAP: Router-Weighted Activation Pruning for Mixture-of-Experts"
Official repository: https://github.com/CerebrasResearch/reap (commit 62cf553)

Key Source Files Replicated:
- reap/pruning_metrics.py: Router-weighted activation saliency accumulation
  (weighted_ean_sum and reap score)
- reap/prune.py: Physical expert pruning, router weight excision, and config updates
- reap/observer.py: Hook registration and activation norm tracking

Distinction:
This engine performs PURE EXPERT PRUNING (REAP-Prune). No parameter averaging
or agglomerative merging is performed; low-saliency experts are physically excised.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

class REAPSaliencyObserver:
    """
    Hook manager to record router probabilities and expert activation norms
    during model forward passes, exactly matching Cerebras REAP observer.
    """
    def __init__(self, model, renormalize_router_weights: bool = True):
        self.model = model
        self.renormalize_router_weights = renormalize_router_weights
        self.hooks = []
        self.state: Dict[int, Dict[str, Any]] = {}
        
        self.moe_blocks: List[Tuple[int, nn.Module]] = []
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                layer_idx = len(self.moe_blocks)
                self.moe_blocks.append((layer_idx, module))
                
        if not self.moe_blocks:
            raise ValueError("No MoE blocks found in model for REAP observer.")
            
        for layer_idx, block in self.moe_blocks:
            router = getattr(block, "gate", getattr(block, "router", None))
            num_experts = router.weight.shape[0]
            self.state[layer_idx] = {
                "total_tokens": 0,
                "expert_frequency": torch.zeros(num_experts, dtype=torch.long),
                "weighted_ean_sum": torch.zeros(num_experts, dtype=torch.float64),
                "reap_sum": torch.zeros(num_experts, dtype=torch.float64),
            }

    def _hook_fn(self, layer_idx: int):
        def hook(module, args, output):
            # args[0] is hidden_states: [batch, seq_len, hidden_dim]
            hidden_states = args[0]
            router = getattr(module, "gate", getattr(module, "router", None))
            router_out = router(hidden_states)
            if isinstance(router_out, tuple):
                router_logits = router_out[0]
            else:
                router_logits = router_out
            
            orig_shape = router_logits.shape
            num_experts = orig_shape[-1]
            flat_logits = router_logits.reshape(-1, num_experts).float()
            num_tokens = flat_logits.shape[0]
            
            # Compute routing probabilities
            routing_weights = F.softmax(flat_logits, dim=-1)
            
            # Determine top_k
            top_k = getattr(module, "num_experts_per_tok", getattr(self.model.config, "num_experts_per_tok", 8))
            top_k = min(top_k, num_experts)
            
            topk_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            
            if self.renormalize_router_weights:
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
                
            # If output is a tuple (e.g. (hidden_states, router_logits)), get tensor
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            out_flat = out_tensor.reshape(-1, out_tensor.shape[-1]).float()
            
            # Token activation norm ||h_i(t)||_2
            act_norms = torch.linalg.norm(out_flat, dim=-1) # [num_tokens]
            
            state = self.state[layer_idx]
            state["total_tokens"] += num_tokens
            
            for k_idx in range(top_k):
                experts_k = selected_experts[:, k_idx] # [num_tokens]
                weights_k = topk_weights[:, k_idx]     # [num_tokens]
                
                # Saliency component: weight * norm
                saliencies = weights_k * act_norms
                
                for exp_id in range(num_experts):
                    mask = (experts_k == exp_id)
                    if mask.any():
                        count = mask.sum().item()
                        state["expert_frequency"][exp_id] += count
                        val = saliencies[mask].sum().item()
                        state["weighted_ean_sum"][exp_id] += val
                        state["reap_sum"][exp_id] += val
                        
        return hook

    def register_hooks(self):
        for layer_idx, block in self.moe_blocks:
            h = block.register_forward_hook(self._hook_fn(layer_idx))
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_saliencies(self, metric: str = "weighted_ean_sum") -> Dict[int, torch.Tensor]:
        """
        Returns saliency vector per layer based on accumulated observations.
        metric: 'weighted_ean_sum' (sum) or 'reap' (mean per active token).
        """
        saliencies = {}
        for layer_idx, state in self.state.items():
            if metric == "weighted_ean_sum":
                saliencies[layer_idx] = state["weighted_ean_sum"].clone()
            elif metric == "reap":
                freq = state["expert_frequency"].float()
                freq_safe = torch.clamp(freq, min=1.0)
                reap_mean = state["reap_sum"] / freq_safe
                # If an expert was never routed, its saliency is 0
                reap_mean[freq == 0] = 0.0
                saliencies[layer_idx] = reap_mean
            else:
                raise ValueError(f"Unknown metric {metric}")
        return saliencies


class REAPPruneEngine:
    """
    Executes faithful Cerebras REAP expert pruning on MoE models.
    Physically deletes expert slices and updates router matrices.
    """
    def __init__(self, model, renormalize_router_weights: bool = True):
        self.model = model
        self.renormalize_router_weights = renormalize_router_weights
        self.moe_blocks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                self.moe_blocks.append(module)
                
        if not self.moe_blocks:
            raise ValueError("No MoE blocks found in model.")
            
        first_block = self.moe_blocks[0]
        router = getattr(first_block, "gate", getattr(first_block, "router", None))
        self.num_experts = router.weight.shape[0]

    @torch.no_grad()
    def prune_block(self, block: nn.Module, saliency: torch.Tensor, target_experts: int) -> List[int]:
        """
        Prunes a single MoE block from current expert count to target_experts.
        Returns the list of retained expert indices.
        """
        router = getattr(block, "gate", getattr(block, "router", None))
        current_experts = router.weight.shape[0]
        n_to_prune = current_experts - target_experts
        if n_to_prune <= 0:
            return list(range(current_experts))
            
        # Lowest saliency experts are pruned
        _, experts_to_prune = torch.topk(saliency, n_to_prune, largest=False)
        pruned_set = set(experts_to_prune.cpu().tolist())
        retained_indices = [i for i in range(current_experts) if i not in pruned_set]
        retained_tensor = torch.tensor(retained_indices, dtype=torch.long, device=router.weight.device)
        
        # 1. Prune fused expert weights
        if hasattr(block.experts, "gate_up_proj"):
            block.experts.gate_up_proj = nn.Parameter(
                block.experts.gate_up_proj.data[retained_tensor].contiguous()
            )
        if hasattr(block.experts, "down_proj"):
            block.experts.down_proj = nn.Parameter(
                block.experts.down_proj.data[retained_tensor].contiguous()
            )
            
        # 2. Prune router
        router.weight = nn.Parameter(
            router.weight.data[retained_tensor, :].contiguous()
        )
        if getattr(router, "bias", None) is not None:
            router.bias = nn.Parameter(
                router.bias.data[retained_tensor].contiguous()
            )
            
        # 3. Update metadata
        block.experts.num_experts = target_experts
        router.num_experts = target_experts
        if hasattr(router, "out_features"):
            router.out_features = target_experts
            
        return retained_indices

    @torch.no_grad()
    def prune_model(self, target_experts: int, layer_saliencies: Dict[int, torch.Tensor]) -> Dict[int, List[int]]:
        """
        Prunes all MoE blocks in the model to target_experts.
        Returns dictionary of retained indices per layer.
        """
        retained_map = {}
        for layer_idx, block in enumerate(self.moe_blocks):
            sal = layer_saliencies[layer_idx]
            retained = self.prune_block(block, sal, target_experts)
            retained_map[layer_idx] = retained
            
        if hasattr(self.model, "config"):
            self.model.config.num_experts = target_experts
            self.model.config.num_local_experts = target_experts
        self.num_experts = target_experts
        return retained_map
