import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# -------------------------------------------------------------------
# HC-SMoE Clustering Core (Faithful to upstream ICML 2025 algorithm)
# -------------------------------------------------------------------

@torch.no_grad()
def pairwise_distances(X: torch.Tensor, method: str = 'average') -> torch.Tensor:
    """Compute pairwise Euclidean distances between expert representations."""
    dot_product = torch.mm(X, X.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0).sqrt()
    distances.fill_diagonal_(float('inf'))
    return distances

@torch.no_grad()
def compute_cluster_distance(pair_distances: torch.Tensor, clusters: torch.Tensor, method: str = 'average', X: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    cluster_labels = torch.unique(clusters)
    num_clusters = len(cluster_labels)
    distances = torch.full((num_clusters, num_clusters), float('inf'), device=clusters.device)
    
    if method == 'average':
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                members_i = torch.where(clusters == ci)[0]
                members_j = torch.where(clusters == cj)[0]
                dist_sub = pair_distances[members_i][:, members_j]
                distances[i, j] = dist_sub.mean()
                distances[j, i] = distances[i, j]
                
    elif method == 'ward':
        cluster_centers = []
        for c in cluster_labels:
            members = torch.where(clusters == c)[0]
            cluster_centers.append(X[members].mean(dim=0))
        cluster_centers = torch.stack(cluster_centers)
        
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                ni = torch.sum(clusters == ci).float()
                nj = torch.sum(clusters == cj).float()
                diff = cluster_centers[i] - cluster_centers[j]
                dist_val = (ni * nj) / (ni + nj) * torch.norm(diff)
                distances[i, j] = dist_val
                distances[j, i] = dist_val
    else:
        raise NotImplementedError(f"Unsupported linkage method: {method}")
        
    distances.fill_diagonal_(float('inf'))
    min_idx = torch.argmin(distances).item()
    row = min_idx // num_clusters
    col = min_idx % num_clusters
    return cluster_labels[row].item(), cluster_labels[col].item()

@torch.no_grad()
def hierarchical_clustering_experts(expert_representations: torch.Tensor, target_k: int, linkage: str = 'average') -> Tuple[List[int], torch.Tensor]:
    """
    Hierarchical agglomerative clustering on expert output representations down to target_k clusters.
    Returns:
      dominant_experts: List of expert indices representing cluster centroids
      cluster_assignments: Tensor of cluster label per original expert
    """
    n_experts = expert_representations.shape[0]
    pair_distances = pairwise_distances(expert_representations, method=linkage)
    clusters = torch.arange(n_experts, device=expert_representations.device)
    
    while len(torch.unique(clusters)) > target_k:
        c_i, c_j = compute_cluster_distance(pair_distances, clusters, method=linkage, X=expert_representations)
        if c_i > c_j:
            c_i, c_j = c_j, c_i
        # Merge cluster j into cluster i
        clusters[clusters == c_j] = c_i

    # Relabel clusters to 0 .. target_k - 1
    unique_clusters = torch.unique(clusters)
    cluster_assignments = torch.zeros_like(clusters)
    dominant_experts = []
    
    for new_label, c in enumerate(unique_clusters):
        members = torch.where(clusters == c)[0]
        cluster_assignments[members] = new_label
        
        # Dominant expert is closest to the cluster center in representation space
        if len(members) == 1:
            dominant_experts.append(members[0].item())
        else:
            center = expert_representations[members].mean(dim=0, keepdim=True)
            dists = torch.norm(expert_representations[members] - center, dim=1)
            closest_local_idx = torch.argmin(dists).item()
            dominant_experts.append(members[closest_local_idx].item())
            
    return dominant_experts, cluster_assignments


# -------------------------------------------------------------------
# Architecture Adapters: JetMoE-8B & Phi-3.5-MoE
# -------------------------------------------------------------------

@torch.no_grad()
def apply_hcsmoe_jetmoe_layer(mlp_module, target_k: int, linkage: str = "average", usage_freq: Optional[torch.Tensor] = None):
    """
    Merges JetMoE parallel expert tensors using HC-SMoE output-based hierarchical clustering.
    mlp_module contains:
      - input_linear.weight: [N, 11264, 2048]
      - output_linear.weight: [N, 2048, 5632]
      - router.layer.weight: [N, 2048]
    """
    input_w = mlp_module.input_linear.weight.data
    output_w = mlp_module.output_linear.weight.data
    router_w = mlp_module.router.layer.weight.data
    n_exp = input_w.shape[0]
    
    # Expert output representation proxy: input projection row mean + output projection column mean
    # Captures functional transformation of the expert
    rep1 = input_w.mean(dim=1)  # [N, 2048]
    rep2 = output_w.mean(dim=2) # [N, 2048]
    expert_reps = (rep1 + rep2) / 2.0
    expert_reps = F.normalize(expert_reps, p=2, dim=1)
    
    dom_experts, assignments = hierarchical_clustering_experts(expert_reps, target_k, linkage=linkage)
    
    dev = input_w.device
    new_input = torch.zeros((target_k, input_w.shape[1], input_w.shape[2]), dtype=input_w.dtype, device=dev)
    new_output = torch.zeros((target_k, output_w.shape[1], output_w.shape[2]), dtype=output_w.dtype, device=dev)
    new_router = torch.zeros((target_k, router_w.shape[1]), dtype=router_w.dtype, device=dev)
    
    for k in range(target_k):
        members = torch.where(assignments == k)[0]
        if usage_freq is not None:
            w = usage_freq[members].float()
            if w.sum() > 0:
                weights = w / w.sum()
            else:
                weights = torch.ones(len(members), device=dev) / len(members)
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
        
    return assignments, dom_experts


def _is_4bit(param) -> bool:
    return hasattr(param, "quant_state")

def _dequantize_to_float(linear: nn.Module, device) -> torch.Tensor:
    w = linear.weight
    if _is_4bit(w):
        try:
            import bitsandbytes as bnb
            return bnb.functional.dequantize_4bit(
                w.data, w.quant_state
            ).to(torch.float32).to(device)
        except Exception:
            pass
    return w.data.float().to(device)

def _requantize_inplace(linear: nn.Module, merged_fp: torch.Tensor) -> None:
    w = linear.weight
    if _is_4bit(w):
        try:
            import bitsandbytes as bnb
            qs = w.quant_state
            blocksize = getattr(qs, "blocksize", 64)
            quant_type = getattr(qs, "quant_type", "fp4")
            new_packed, new_qs = bnb.functional.quantize_4bit(
                merged_fp.to(torch.float16),
                blocksize=blocksize,
                quant_type=quant_type,
            )
            w.data = new_packed
            w.quant_state = new_qs
            return
        except Exception:
            pass
    target_dtype = getattr(w, "dtype", torch.bfloat16)
    if str(target_dtype).startswith("torch.int") or str(target_dtype).startswith("torch.uint"):
        target_dtype = torch.bfloat16
    linear.weight = nn.Parameter(merged_fp.to(target_dtype))

@torch.no_grad()
def apply_hcsmoe_phi_layer(sparse_moe_module, target_k: int, linkage: str = "average", usage_freq: Optional[torch.Tensor] = None):
    """
    Merges Phi-3.5-MoE expert tensors using HC-SMoE output-based hierarchical clustering.
    Supports both:
      - ModuleList layout: sparse_moe_module.experts with w1, w2, w3 and sparse_moe_module.gate
      - Batched layout: sparse_moe_module.gate_up_proj, down_proj, gate
    """
    if hasattr(sparse_moe_module, "experts") and isinstance(sparse_moe_module.experts, (list, nn.ModuleList)):
        # Official Phi-3.5-MoE architecture (nn.ModuleList of experts with w1, w2, w3)
        device = next(iter(sparse_moe_module.parameters())).device
        n_exp = len(sparse_moe_module.experts)
        
        reps = []
        for e in sparse_moe_module.experts:
            w1_f = _dequantize_to_float(e.w1, device)
            r1 = w1_f.mean(dim=0)
            del w1_f
            w2_f = _dequantize_to_float(e.w2, device)
            r2 = w2_f.mean(dim=1)
            del w2_f
            r = F.normalize((r1 + r2) / 2.0, p=2, dim=0)
            reps.append(r)
        expert_reps = torch.stack(reps)
        
        dom_experts, assignments = hierarchical_clustering_experts(expert_reps, target_k, linkage=linkage)
        
        # Merge non-dominant into dominant experts (sequential accumulation to bound VRAM)
        for k in range(target_k):
            members = torch.where(assignments == k)[0].tolist()
            dom = dom_experts[k]
            for attr in ['w1', 'w2', 'w3']:
                dom_lin = getattr(sparse_moe_module.experts[dom], attr)
                merged = None
                for m in members:
                    m_f = _dequantize_to_float(getattr(sparse_moe_module.experts[m], attr), device)
                    if merged is None:
                        merged = m_f
                    else:
                        merged += m_f
                    del m_f
                merged /= len(members)
                _requantize_inplace(dom_lin, merged)
                del merged
                
        # Update router
        gate_w = _dequantize_to_float(sparse_moe_module.gate, device)
        new_gate_w = torch.zeros((target_k, gate_w.shape[1]), dtype=gate_w.dtype, device=device)
        for k in range(target_k):
            members = torch.where(assignments == k)[0].tolist()
            new_gate_w[k] = sum(gate_w[m] for m in members) / len(members)
            
        gate_bias = getattr(sparse_moe_module.gate, "bias", None)
        new_gate = nn.Linear(gate_w.shape[1], target_k, bias=gate_bias is not None, device=device)
        new_gate.weight = nn.Parameter(new_gate_w.to(torch.bfloat16))
        if gate_bias is not None:
            new_gate_bias = torch.zeros((target_k,), dtype=gate_bias.dtype, device=device)
            for k in range(target_k):
                members = torch.where(assignments == k)[0].tolist()
                new_gate_bias[k] = sum(gate_bias.data[m] for m in members) / len(members)
            new_gate.bias = nn.Parameter(new_gate_bias.to(torch.bfloat16))
        sparse_moe_module.gate = new_gate
        del gate_w, new_gate_w
        
        # Prune experts ModuleList
        sparse_moe_module.experts = nn.ModuleList([sparse_moe_module.experts[dom] for dom in dom_experts])
        sparse_moe_module.num_experts = target_k
        if hasattr(sparse_moe_module, "num_local_experts"):
            sparse_moe_module.num_local_experts = target_k
        torch.cuda.empty_cache()
            
        return assignments, dom_experts
        
    elif hasattr(sparse_moe_module, "gate_up_proj"):
        gate_up_w = sparse_moe_module.gate_up_proj.data
        down_w = sparse_moe_module.down_proj.data
        router_w = sparse_moe_module.gate.weight.data
        n_exp = gate_up_w.shape[0]
        
        rep1 = gate_up_w.mean(dim=1)
        rep2 = down_w.mean(dim=2)
        expert_reps = (rep1 + rep2) / 2.0
        expert_reps = F.normalize(expert_reps, p=2, dim=1)
        
        dom_experts, assignments = hierarchical_clustering_experts(expert_reps, target_k, linkage=linkage)
        
        dev = gate_up_w.device
        new_gate_up = torch.zeros((target_k, gate_up_w.shape[1], gate_up_w.shape[2]), dtype=gate_up_w.dtype, device=dev)
        new_down = torch.zeros((target_k, down_w.shape[1], down_w.shape[2]), dtype=down_w.dtype, device=dev)
        new_router = torch.zeros((target_k, router_w.shape[1]), dtype=router_w.dtype, device=dev)
        
        for k in range(target_k):
            members = torch.where(assignments == k)[0]
            if usage_freq is not None:
                w = usage_freq[members].float()
                if w.sum() > 0:
                    weights = w / w.sum()
                else:
                    weights = torch.ones(len(members), device=dev) / len(members)
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
            
        return assignments, dom_experts
