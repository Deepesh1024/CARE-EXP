import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Tuple, Dict, Any

class MoECompressionEngine:
    def __init__(self, model):
        self.model = model
        # Locate ALL MoE blocks
        self.moe_blocks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                self.moe_blocks.append(module)
                
        if not self.moe_blocks:
            raise ValueError("Could not locate any MoE blocks in the model.")
            
        # Assume all layers have the same number of experts
        first_block = self.moe_blocks[0]
        router = getattr(first_block, "gate", getattr(first_block, "router", None))
        self.current_num_experts = router.weight.shape[0]
        self.active_experts = list(range(self.current_num_experts))

    def get_expert_params(self, expert_idx: int):
        """Returns flattened parameters for a specific expert across ALL layers."""
        all_params = []
        for block in self.moe_blocks:
            # FIX 1: OLMoE has separate gate, up, and down projections
            w1 = block.experts[expert_idx].gate_proj.weight.detach().cpu().view(-1)
            w2 = block.experts[expert_idx].up_proj.weight.detach().cpu().view(-1)
            w3 = block.experts[expert_idx].down_proj.weight.detach().cpu().view(-1)
            all_params.extend([w1, w2, w3])
        return torch.cat(all_params)

    @torch.no_grad()
    def merge_experts(self, expert_i: int, expert_j: int):
        """
        Physically merges expert_j into expert_i by averaging their weights across ALL layers.
        The tensor size is reduced by 1.
        """
        if expert_i not in self.active_experts or expert_j not in self.active_experts:
            raise ValueError(f"One of experts {expert_i} or {expert_j} is not active.")

        mapping = {}
        for block in self.moe_blocks:
            router_module = getattr(block, "gate", getattr(block, "router", None))
            
            # FIX 1: Merge separate Expert Weights (Averaging)
            gate_new = (block.experts[expert_i].gate_proj.weight.data + block.experts[expert_j].gate_proj.weight.data) / 2.0
            up_new = (block.experts[expert_i].up_proj.weight.data + block.experts[expert_j].up_proj.weight.data) / 2.0
            down_new = (block.experts[expert_i].down_proj.weight.data + block.experts[expert_j].down_proj.weight.data) / 2.0
            
            # 2. Merge Router Weights
            router_w = router_module.weight.data
            router_new = (router_w[expert_i] + router_w[expert_j]) / 2.0
            
            # Replace weights in expert_i
            block.experts[expert_i].gate_proj.weight.data.copy_(gate_new)
            block.experts[expert_i].up_proj.weight.data.copy_(up_new)
            block.experts[expert_i].down_proj.weight.data.copy_(down_new)
            
            # 3. Create new smaller ModuleList by excluding expert_j
            keep_indices = [idx for idx in range(len(block.experts)) if idx != expert_j]
            new_experts = nn.ModuleList([block.experts[idx] for idx in keep_indices])
            block.experts = new_experts
            
            new_expert_i = keep_indices.index(expert_i)
            
            # Update Router
            new_router_w = router_w[keep_indices].clone()
            new_router_w[new_expert_i] = router_new
            router_module.weight = nn.Parameter(new_router_w)
            
            new_num_experts = len(keep_indices)
            
            # FIX 2: HuggingFace stores `num_experts` on the block and router!
            # We must recursively update it so the forward loop doesn't iterate out of bounds (fixes Index 56 error).
            for module in block.modules():
                if hasattr(module, 'num_experts'):
                    module.num_experts = new_num_experts
            
            # Create mapping
            mapping = {}
            for old_idx, new_idx in zip(keep_indices, range(new_num_experts)):
                if old_idx == expert_i:
                    mapping[old_idx] = new_expert_i
                else:
                    mapping[old_idx] = new_idx
            mapping[expert_j] = new_expert_i

        # Update config attributes
        if hasattr(self.model.config, 'num_experts'):
            self.model.config.num_experts = new_num_experts
            
        self.current_num_experts = new_num_experts
        self.active_experts = list(range(self.current_num_experts))
        
        return mapping

def greedy_conflict_resolution(candidate_pairs: List[Tuple[int, int, float]], target_merges: int) -> List[Tuple[int, int]]:
    """
    candidates_pairs: list of (expert_i, expert_j, score) sorted by score (best first).
    Returns a list of exactly `target_merges` valid pairs to merge.
    """
    accepted_merges = []
    locked_experts = set()
    
    for i, j, score in candidate_pairs:
        if len(accepted_merges) >= target_merges:
            break
        if i not in locked_experts and j not in locked_experts:
            accepted_merges.append((i, j))
            locked_experts.add(i)
            locked_experts.add(j)
            
    return accepted_merges
