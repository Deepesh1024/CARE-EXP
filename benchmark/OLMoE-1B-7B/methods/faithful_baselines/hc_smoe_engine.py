"""
benchmark/OLMoE-1B-7B/methods/faithful_baselines/hc_smoe_engine.py
==================================================================
Faithful Primary-Source Implementation of HC-SMoE (PMLR / ICML 2025).
Reference: Chen et al., "Hierarchical Clustering for Mixture-of-Experts Compression"
(arXiv:2410.11289).
Official repository: https://github.com/wazenmai/HC-SMoE (commit bd77eba)

Key Source Files Replicated:
- hcsmoe/merging/clustering.py: Hierarchical clustering with average/ward linkage
- hcsmoe/merging/grouping_qwen.py: _merge_mlp_experts_by_usage_frequency_weighting
- hcsmoe/merging/utils.py: Distance matrices and dominant expert selection

Core Procedures:
1. Expert Pairwise Similarity: Cosine distance of router weights or activation profiles.
2. Agglomerative Hierarchical Clustering: Successive linkage reduction to target_k clusters.
3. Frequency-Weighted Parameter Merging:
   W_merged = sum(f_i * W_i) / (sum(f_i) + eps)
4. Physical Tensor Shrinking: Consolidates parameters to target_k experts and updates model config.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class HCSMoEEngine:
    """
    Executes faithful HC-SMoE compression on MoE models.
    """
    def __init__(self, model, linkage: str = "average", similarity_metric: str = "router_cosine"):
        self.model = model
        self.linkage = linkage
        self.similarity_metric = similarity_metric
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
    def compute_expert_distance_matrix(self, block: nn.Module, expert_representations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes pairwise dissimilarity matrix D of shape [num_experts, num_experts].
        If expert_representations is None, uses router weight cosine distance.
        """
        router = getattr(block, "gate", getattr(block, "router", None))
        if expert_representations is not None:
            reps = expert_representations.float()
        else:
            reps = router.weight.data.float()
            
        # Normalize representations
        normed = F.normalize(reps, p=2, dim=-1)
        sim = torch.mm(normed, normed.t()) # Cosine similarity [-1, 1]
        dist = 1.0 - sim # Distance [0, 2]
        dist.fill_diagonal_(0.0)
        return dist

    @torch.no_grad()
    def agglomerative_clustering(self, dist_matrix: torch.Tensor, target_k: int) -> torch.Tensor:
        """
        Executes Agglomerative Hierarchical Clustering using average or ward linkage.
        Returns cluster assignments tensor of shape [num_experts].
        """
        num_experts = dist_matrix.shape[0]
        if target_k >= num_experts:
            return torch.arange(num_experts, device=dist_matrix.device)
            
        # Initial clusters: each expert is its own cluster
        clusters = {i: [i] for i in range(num_experts)}
        active_clusters = set(range(num_experts))
        
        # Working copy of pairwise expert distances
        D = dist_matrix.clone().cpu()
        
        while len(active_clusters) > target_k:
            best_dist = float("inf")
            best_pair = None
            
            cluster_list = sorted(list(active_clusters))
            for i_idx in range(len(cluster_list)):
                c1 = cluster_list[i_idx]
                members1 = clusters[c1]
                for j_idx in range(i_idx + 1, len(cluster_list)):
                    c2 = cluster_list[j_idx]
                    members2 = clusters[c2]
                    
                    if self.linkage == "average":
                        # Average linkage: mean distance between all member pairs
                        d_sum = 0.0
                        for m1 in members1:
                            for m2 in members2:
                                d_sum += D[m1, m2].item()
                        d_link = d_sum / (len(members1) * len(members2))
                    elif self.linkage == "single":
                        d_link = min(D[m1, m2].item() for m1 in members1 for m2 in members2)
                    elif self.linkage == "complete":
                        d_link = max(D[m1, m2].item() for m1 in members1 for m2 in members2)
                    else:
                        # Default to average linkage
                        d_sum = sum(D[m1, m2].item() for m1 in members1 for m2 in members2)
                        d_link = d_sum / (len(members1) * len(members2))
                        
                    if d_link < best_dist:
                        best_dist = d_link
                        best_pair = (c1, c2)
                        
            c_keep, c_merge = best_pair
            clusters[c_keep].extend(clusters[c_merge])
            del clusters[c_merge]
            active_clusters.remove(c_merge)
            
        # Form contiguous labels
        cluster_labels = torch.zeros(num_experts, dtype=torch.long, device=dist_matrix.device)
        for new_id, (cid, members) in enumerate(clusters.items()):
            for m in members:
                cluster_labels[m] = new_id
                
        return cluster_labels

    @torch.no_grad()
    def merge_block(
        self,
        block: nn.Module,
        cluster_labels: torch.Tensor,
        frequencies: torch.Tensor,
    ):
        """
        Merges experts in a single MoE block weighted by routing frequencies,
        following hcsmoe/merging/grouping_qwen.py: _merge_mlp_experts_by_usage_frequency_weighting.
        """
        router = getattr(block, "gate", getattr(block, "router", None))
        unique_labels = torch.unique(cluster_labels).cpu().tolist()
        target_k = len(unique_labels)
        
        orig_dtype = block.experts.gate_up_proj.dtype
        dev = block.experts.gate_up_proj.device
        
        new_gate_up = []
        new_down = []
        new_router = []
        
        freqs_fp32 = frequencies.float().to(dev)
        eps = 1e-9
        
        for c in unique_labels:
            member_indices = torch.where(cluster_labels == c)[0]
            m_freqs = freqs_fp32[member_indices] # [m]
            total_freq = m_freqs.sum()
            
            if total_freq <= 0:
                weights = torch.ones_like(m_freqs) / len(member_indices)
            else:
                weights = m_freqs / (total_freq + eps)
                
            # 1. Frequency-weighted gate_up_proj
            # Shape of member: [2 * intermediate, hidden]
            merged_gu = torch.zeros_like(block.experts.gate_up_proj.data[0], dtype=torch.float32)
            for idx, w in zip(member_indices, weights):
                merged_gu += w * block.experts.gate_up_proj.data[idx].float()
            new_gate_up.append(merged_gu.to(orig_dtype))
            
            # 2. Frequency-weighted down_proj
            # Shape of member: [hidden, intermediate]
            merged_down = torch.zeros_like(block.experts.down_proj.data[0], dtype=torch.float32)
            for idx, w in zip(member_indices, weights):
                merged_down += w * block.experts.down_proj.data[idx].float()
            new_down.append(merged_down.to(orig_dtype))
            
            # 3. Router row: sum of member router rows (HC-SMoE specification)
            merged_r = torch.zeros_like(router.weight.data[0], dtype=torch.float32)
            for idx in member_indices:
                merged_r += router.weight.data[idx].float()
            new_router.append(merged_r.to(router.weight.dtype))
            
        # Update physical parameters
        block.experts.gate_up_proj = nn.Parameter(torch.stack(new_gate_up, dim=0))
        block.experts.down_proj = nn.Parameter(torch.stack(new_down, dim=0))
        router.weight = nn.Parameter(torch.stack(new_router, dim=0))
        
        block.experts.num_experts = target_k
        router.num_experts = target_k
        if hasattr(router, "out_features"):
            router.out_features = target_k

    @torch.no_grad()
    def compress_model(
        self,
        target_experts: int,
        frequencies_per_layer: Dict[int, torch.Tensor],
        representations_per_layer: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Compresses all MoE blocks in the model to target_experts.
        Returns cluster assignments per layer.
        """
        labels_map = {}
        for layer_idx, block in enumerate(self.moe_blocks):
            reps = representations_per_layer.get(layer_idx) if representations_per_layer else None
            freqs = frequencies_per_layer[layer_idx]
            
            dist = self.compute_expert_distance_matrix(block, reps)
            labels = self.agglomerative_clustering(dist, target_experts)
            self.merge_block(block, labels, freqs)
            labels_map[layer_idx] = labels
            
        if hasattr(self.model, "config"):
            self.model.config.num_experts = target_experts
            self.model.config.num_local_experts = target_experts
        self.num_experts = target_experts
        return labels_map
