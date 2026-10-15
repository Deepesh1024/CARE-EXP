"""
care_large_moe_systems/complete_comparative_benchmark_v2/qwen3_clustering_engines.py
Faithful adaptations of HC-SMoE (ICML 2025) and MC-SMoE (UNITES-Lab) for Qwen3-30B-A3B.
Operates directly on Qwen3MoeSparseMoeBlock:
  - block.experts.gate_up_proj: [num_experts, 1536, 2048]
  - block.experts.down_proj:    [num_experts, 2048, 768]
  - block.gate.weight:          [num_experts, 2048]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# =====================================================================
# HC-SMoE Core (Hierarchical Clustering for MoE Compression, ICML 2025)
# Using Lance-Williams Average-Linkage Recurrence (Exact & Millisecond Fast)
# =====================================================================

@torch.no_grad()
def hierarchical_clustering_experts(expert_representations: torch.Tensor, target_k: int) -> Tuple[List[int], torch.Tensor]:
    """
    Hierarchical agglomerative clustering using the exact Lance-Williams average-linkage recurrence.
    Mathematically identical to pairwise member mean distance, but runs in milliseconds.
    """
    N = expert_representations.shape[0]
    D = torch.cdist(expert_representations, expert_representations)
    D.fill_diagonal_(float('inf'))
    
    cluster_members = [[i] for i in range(N)]
    cluster_sizes = torch.ones(N, dtype=torch.float32, device=expert_representations.device)
    active = set(range(N))
    
    while len(active) > target_k:
        active_list = sorted(list(active))
        sub_D = D[active_list][:, active_list]
        min_idx = torch.argmin(sub_D).item()
        r, c = min_idx // len(active_list), min_idx % len(active_list)
        ci, cj = active_list[r], active_list[c]
        if ci > cj:
            ci, cj = cj, ci
            
        # Lance-Williams average linkage update: D(ci U cj, k) = (ni*D(ci,k) + nj*D(cj,k)) / (ni + nj)
        ni, nj = cluster_sizes[ci], cluster_sizes[cj]
        D[ci, :] = (ni * D[ci, :] + nj * D[cj, :]) / (ni + nj)
        D[:, ci] = D[ci, :]
        D[ci, ci] = float('inf')
        
        # Deactivate cj
        D[cj, :] = float('inf')
        D[:, cj] = float('inf')
        cluster_sizes[ci] += nj
        cluster_members[ci].extend(cluster_members[cj])
        active.remove(cj)
        
    cluster_assignments = torch.zeros(N, dtype=torch.long, device=expert_representations.device)
    dominant_experts = []
    
    for new_idx, c in enumerate(sorted(list(active))):
        members = cluster_members[c]
        for m in members:
            cluster_assignments[m] = new_idx
            
        # Dominant expert is closest to cluster center in representation space
        if len(members) == 1:
            dominant_experts.append(members[0])
        else:
            center = expert_representations[members].mean(dim=0, keepdim=True)
            dists = torch.norm(expert_representations[members] - center, dim=1)
            closest_idx = members[torch.argmin(dists).item()]
            dominant_experts.append(closest_idx)
            
    return dominant_experts, cluster_assignments

@torch.no_grad()
def apply_hcsmoe_qwen3_layer(block, target_k: int, usage_freq: Optional[torch.Tensor] = None):
    """
    Applies HC-SMoE clustering to a single Qwen3-30B-A3B MoE layer.
    """
    gate_up_w = block.experts.gate_up_proj.data # [N, 1536, 2048]
    down_w = block.experts.down_proj.data       # [N, 2048, 768]
    router_w = block.gate.weight.data           # [N, 2048]
    n_exp = gate_up_w.shape[0]
    dev = gate_up_w.device
    
    # 1. Output representation proxy: gate_up row mean + down col mean
    rep1 = gate_up_w.mean(dim=1).float()
    rep2 = down_w.mean(dim=2).float()
    expert_reps = F.normalize((rep1 + rep2) / 2.0, p=2, dim=1)
    
    # 2. Hierarchical agglomerative clustering
    dom_experts, assignments = hierarchical_clustering_experts(expert_reps, target_k)
    
    # 3. Parameter merging
    new_gate_up = torch.zeros((target_k, gate_up_w.shape[1], gate_up_w.shape[2]), dtype=gate_up_w.dtype, device=dev)
    new_down = torch.zeros((target_k, down_w.shape[1], down_w.shape[2]), dtype=down_w.dtype, device=dev)
    new_router = torch.zeros((target_k, router_w.shape[1]), dtype=router_w.dtype, device=dev)
    
    for k in range(target_k):
        members = torch.where(assignments == k)[0]
        if usage_freq is not None:
            w = usage_freq[members].float()
            weights = w / w.sum() if w.sum() > 0 else torch.ones(len(members), device=dev) / len(members)
        else:
            weights = torch.ones(len(members), device=dev) / len(members)
            
        merged_gu = torch.sum(gate_up_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_dp = torch.sum(down_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_r = torch.sum(router_w[members] * weights.view(-1, 1), dim=0)
        
        new_gate_up[k] = merged_gu
        new_down[k] = merged_dp
        new_router[k] = merged_r
        
    block.experts.gate_up_proj = nn.Parameter(new_gate_up)
    block.experts.down_proj = nn.Parameter(new_down)
    block.gate.weight = nn.Parameter(new_router)
    block.experts.num_experts = target_k
    block.gate.num_experts = target_k
    return assignments, dom_experts

# =====================================================================
# MC-SMoE / M-SMoE Core (Merging Mixture-of-Experts with Common Representations)
# =====================================================================

@torch.no_grad()
def group_experts_routing_guided(
    usage_frequencies: torch.Tensor,
    similarity_matrix: torch.Tensor,
    target_k: int
) -> Tuple[List[int], torch.Tensor]:
    n_experts = usage_frequencies.shape[0]
    sorted_by_usage = torch.argsort(usage_frequencies, descending=True)
    core_experts = sorted_by_usage[:target_k].tolist()
    
    assignments = torch.zeros(n_experts, dtype=torch.long, device=usage_frequencies.device)
    for new_label, core_idx in enumerate(core_experts):
        assignments[core_idx] = new_label
        
    non_core_experts = sorted_by_usage[target_k:].tolist()
    for non_core in non_core_experts:
        sims_to_cores = similarity_matrix[non_core, core_experts]
        best_core_local_idx = torch.argmax(sims_to_cores).item()
        assignments[non_core] = best_core_local_idx
        
    return core_experts, assignments

@torch.no_grad()
def apply_mcsmoe_qwen3_layer(
    block,
    target_k: int,
    usage_freq: Optional[torch.Tensor] = None
):
    """
    Applies MC-SMoE / M-SMoE clustering to a single Qwen3-30B-A3B MoE layer.
    """
    gate_up_w = block.experts.gate_up_proj.data # [N, 1536, 2048]
    down_w = block.experts.down_proj.data       # [N, 2048, 768]
    router_w = block.gate.weight.data           # [N, 2048]
    n_exp = gate_up_w.shape[0]
    dev = gate_up_w.device
    
    # 1. Expert usage frequency
    if usage_freq is None:
        usage_freq = router_w.norm(dim=1)
        usage_freq = usage_freq / usage_freq.sum()
        
    # 2. Router cosine similarity matrix
    normed_router = F.normalize(router_w.float(), p=2, dim=1)
    sim_matrix = torch.mm(normed_router, normed_router.t())
    
    # 3. Core selection and routing-guided assignment
    core_experts, assignments = group_experts_routing_guided(usage_freq, sim_matrix, target_k)
    
    # 4. Parameter merging
    new_gate_up = torch.zeros((target_k, gate_up_w.shape[1], gate_up_w.shape[2]), dtype=gate_up_w.dtype, device=dev)
    new_down = torch.zeros((target_k, down_w.shape[1], down_w.shape[2]), dtype=down_w.dtype, device=dev)
    new_router = torch.zeros((target_k, router_w.shape[1]), dtype=router_w.dtype, device=dev)
    
    for k in range(target_k):
        members = torch.where(assignments == k)[0]
        w = usage_freq[members].float()
        weights = w / w.sum() if w.sum() > 0 else torch.ones(len(members), device=dev) / len(members)
        
        merged_gu = torch.sum(gate_up_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_dp = torch.sum(down_w[members] * weights.view(-1, 1, 1), dim=0)
        merged_r = torch.sum(router_w[members] * weights.view(-1, 1), dim=0)
        
        new_gate_up[k] = merged_gu
        new_down[k] = merged_dp
        new_router[k] = merged_r
        
    block.experts.gate_up_proj = nn.Parameter(new_gate_up)
    block.experts.down_proj = nn.Parameter(new_down)
    block.gate.weight = nn.Parameter(new_router)
    block.experts.num_experts = target_k
    block.gate.num_experts = target_k
    return assignments, core_experts
