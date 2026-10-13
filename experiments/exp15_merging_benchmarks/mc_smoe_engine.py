import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# -------------------------------------------------------------------
# MC-SMoE / M-SMoE Merging Core (Faithful to upstream UNITES-Lab logic)
# -------------------------------------------------------------------

@torch.no_grad()
def group_experts_routing_guided(
    usage_frequencies: torch.Tensor,
    similarity_matrix: torch.Tensor,
    target_k: int
) -> Tuple[List[int], torch.Tensor]:
    """
    Implements M-SMoE (Stage 1 of MC-SMoE):
    1. Select top-K experts with highest usage frequency as dominant core experts.
    2. Assign each non-core expert to the most similar core expert based on similarity_matrix.
    Returns:
      core_experts: list of K core expert indices
      assignments: cluster label (0 .. target_k - 1) for each original expert
    """
    n_experts = usage_frequencies.shape[0]
    sorted_by_usage = torch.argsort(usage_frequencies, descending=True)
    core_experts = sorted_by_usage[:target_k].tolist()
    
    assignments = torch.zeros(n_experts, dtype=torch.long, device=usage_frequencies.device)
    for new_label, core_idx in enumerate(core_experts):
        assignments[core_idx] = new_label
        
    non_core_experts = sorted_by_usage[target_k:].tolist()
    for non_core in non_core_experts:
        # Cosine similarity to each core expert
        sims_to_cores = similarity_matrix[non_core, core_experts]
        best_core_local_idx = torch.argmax(sims_to_cores).item()
        assignments[non_core] = best_core_local_idx
        
    return core_experts, assignments

# -------------------------------------------------------------------
# Architecture Adapters: JetMoE-8B & Phi-3.5-MoE
# -------------------------------------------------------------------

@torch.no_grad()
def apply_mcsmoe_jetmoe_layer(
    mlp_module,
    target_k: int,
    usage_freq: Optional[torch.Tensor] = None
):
    """
    Merges JetMoE parallel expert tensors using M-SMoE routing-guided grouping.
    mlp_module contains:
      - input_linear.weight: [N, 11264, 2048]
      - output_linear.weight: [N, 2048, 5632]
      - router.layer.weight: [N, 2048]
    """
    input_w = mlp_module.input_linear.weight.data
    output_w = mlp_module.output_linear.weight.data
    router_w = mlp_module.router.layer.weight.data
    n_exp = input_w.shape[0]
    dev = input_w.device
    
    # 1. Usage frequencies
    if usage_freq is None:
        # Fallback to router weight norm as proxy if calibration profile not provided
        usage_freq = router_w.norm(dim=1)
        usage_freq = usage_freq / usage_freq.sum()
        
    # 2. Similarity matrix from router projection weights (upstream MC-SMoE router-logits/weights metric)
    normed_router = F.normalize(router_w.float(), p=2, dim=1)
    sim_matrix = torch.mm(normed_router, normed_router.t())
    
    core_experts, assignments = group_experts_routing_guided(usage_freq, sim_matrix, target_k)
    
    new_input = torch.zeros((target_k, input_w.shape[1], input_w.shape[2]), dtype=input_w.dtype, device=dev)
    new_output = torch.zeros((target_k, output_w.shape[1], output_w.shape[2]), dtype=output_w.dtype, device=dev)
    new_router = torch.zeros((target_k, router_w.shape[1]), dtype=router_w.dtype, device=dev)
    
    for k in range(target_k):
        members = torch.where(assignments == k)[0]
        w = usage_freq[members].float()
        if w.sum() > 0:
            weights = w / w.sum()
        else:
            weights = torch.ones(len(members), device=dev) / len(members)
            
        merged_in = torch.sum(input_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_out = torch.sum(output_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_r = torch.sum(router_w[members] * weights.view(-1, 1), dim=0)
        
        new_input[k] = merged_in
        new_output[k] = merged_out
        new_router[k] = merged_r
        
    mlp_module.input_linear.weight = nn.Parameter(new_input)
    mlp_module.output_linear.weight = nn.Parameter(new_output)
    mlp_module.router.layer.weight = nn.Parameter(new_router)
    if hasattr(mlp_module, "input_linear") and hasattr(mlp_module.input_linear, "num_experts"):
        mlp_module.input_linear.num_experts = target_k
    if hasattr(mlp_module, "output_linear") and hasattr(mlp_module.output_linear, "num_experts"):
        mlp_module.output_linear.num_experts = target_k
    if hasattr(mlp_module, "router") and hasattr(mlp_module.router, "num_experts"):
        mlp_module.router.num_experts = target_k
    if hasattr(mlp_module, "num_experts"):
        mlp_module.num_experts = target_k
        
    return assignments, core_experts


@torch.no_grad()
def apply_mcsmoe_phi_layer(
    sparse_moe_module,
    target_k: int,
    usage_freq: Optional[torch.Tensor] = None
):
    """
    Merges Phi-3.5-MoE expert tensors using M-SMoE routing-guided grouping.
    sparse_moe_module contains:
      - gate_up_proj: [N, 12800, 4096]
      - down_proj: [N, 4096, 6400]
      - gate: Linear(4096, N)
    """
    gate_up_w = sparse_moe_module.gate_up_proj.data
    down_w = sparse_moe_module.down_proj.data
    router_w = sparse_moe_module.gate.weight.data
    n_exp = gate_up_w.shape[0]
    dev = gate_up_w.device
    
    if usage_freq is None:
        usage_freq = router_w.norm(dim=1)
        usage_freq = usage_freq / usage_freq.sum()
        
    normed_router = F.normalize(router_w.float(), p=2, dim=1)
    sim_matrix = torch.mm(normed_router, normed_router.t())
    
    core_experts, assignments = group_experts_routing_guided(usage_freq, sim_matrix, target_k)
    
    new_gate_up = torch.zeros((target_k, gate_up_w.shape[1], gate_up_w.shape[2]), dtype=gate_up_w.dtype, device=dev)
    new_down = torch.zeros((target_k, down_w.shape[1], down_w.shape[2]), dtype=down_w.dtype, device=dev)
    new_router = torch.zeros((target_k, router_w.shape[1]), dtype=router_w.dtype, device=dev)
    
    for k in range(target_k):
        members = torch.where(assignments == k)[0]
        w = usage_freq[members].float()
        if w.sum() > 0:
            weights = w / w.sum()
        else:
            weights = torch.ones(len(members), device=dev) / len(members)
            
        merged_gu = torch.sum(gate_up_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_down = torch.sum(down_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_r = torch.sum(router_w[members] * weights.view(-1, 1), dim=0)
        
        new_gate_up[k] = merged_gu
        new_down[k] = merged_down
        new_router[k] = merged_r
        
    sparse_moe_module.gate_up_proj = nn.Parameter(new_gate_up)
    sparse_moe_module.down_proj = nn.Parameter(new_down)
    sparse_moe_module.gate.weight = nn.Parameter(new_router)
    if hasattr(sparse_moe_module, "num_experts"):
        sparse_moe_module.num_experts = target_k
        
    return assignments, core_experts
