"""
benchmark/OLMoE-1B-7B/methods/faithful_baselines/mc_smoe_engine.py
==================================================================
Faithful Primary-Source Implementation of MC-SMoE (OpenReview / ICLR 2024).
Reference: Li et al., "MC-SMoE: Merging and Compressing Mixture-of-Experts"
(arXiv:2402.00396).
Official repository: https://github.com/UNITES-Lab/MC-SMoE (commit 99b400f)

Key Source Files Replicated:
- mcsmoe/merging/permutation.py: Hungarian bipartite matching on hidden neurons
  via scipy.optimize.linear_sum_assignment
- mcsmoe/merging/grouping.py: Routing-policy similarity grouping
- mcsmoe/merging/grouping_mixtral.py: Frequency-weighted consolidation

Core Procedures:
1. Routing-Policy Clustering: Groups experts based on router weight cosine similarity.
2. Dominant Expert Identification: Highest frequency expert in each cluster serves as reference frame.
3. Hungarian Bipartite Neuron Matching: Solves optimal intermediate neuron permutation:
   Cost_{u, v} = <W_gate_r[u], W_gate_j[v]> + <W_up_r[u], W_up_j[v]> + <W_down_r[:, u], W_down_j[:, v]>
   pi* = linear_sum_assignment(-Cost)
4. Permutation-Aligned Weight Merging: Permutes expert j's intermediate channels into expert r's
   frame, followed by activation-frequency-weighted averaging:
   W_merged = sum(f_i * W_i^(pi)) / (sum(f_i) + eps)
5. Physical Tensor Shrinking: Excises redundant slots and updates router.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import linear_sum_assignment

class MCSMoEEngine:
    """
    Executes faithful MC-SMoE compression with Hungarian neuron permutation alignment.
    """
    def __init__(self, model):
        self.model = model
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
    def cluster_experts_by_router_policy(self, block: nn.Module, target_k: int) -> torch.Tensor:
        """
        Groups experts into target_k clusters based on router weight cosine similarity.
        """
        router = getattr(block, "gate", getattr(block, "router", None))
        reps = router.weight.data.float()
        normed = F.normalize(reps, p=2, dim=-1)
        sim = torch.mm(normed, normed.t()) # [num_experts, num_experts]
        dist = 1.0 - sim
        dist.fill_diagonal_(0.0)
        
        num_experts = reps.shape[0]
        clusters = {i: [i] for i in range(num_experts)}
        active = set(range(num_experts))
        D = dist.clone().cpu()
        
        while len(active) > target_k:
            best_dist = float("inf")
            best_pair = None
            active_list = sorted(list(active))
            for i_idx in range(len(active_list)):
                c1 = active_list[i_idx]
                m1 = clusters[c1]
                for j_idx in range(i_idx + 1, len(active_list)):
                    c2 = active_list[j_idx]
                    m2 = clusters[c2]
                    d_mean = sum(D[u, v].item() for u in m1 for v in m2) / (len(m1) * len(m2))
                    if d_mean < best_dist:
                        best_dist = d_mean
                        best_pair = (c1, c2)
            c_keep, c_merge = best_pair
            clusters[c_keep].extend(clusters[c_merge])
            del clusters[c_merge]
            active.remove(c_merge)
            
        labels = torch.zeros(num_experts, dtype=torch.long, device=reps.device)
        for new_id, (cid, members) in enumerate(clusters.items()):
            for m in members:
                labels[m] = new_id
        return labels

    @torch.no_grad()
    def compute_bipartite_neuron_matching(
        self,
        ref_gate: torch.Tensor,
        ref_up: torch.Tensor,
        ref_down: torch.Tensor,
        target_gate: torch.Tensor,
        target_up: torch.Tensor,
        target_down: torch.Tensor
    ) -> torch.Tensor:
        """
        Solves the Hungarian bipartite matching between intermediate neurons of
        reference expert and target expert.
        
        Args:
            ref_gate, target_gate: [d_ff, d_model]
            ref_up, target_up: [d_ff, d_model]
            ref_down, target_down: [d_model, d_ff]
            
        Returns:
            perm: torch.Tensor of shape [d_ff], where perm[u] is the index of the
                  target neuron assigned to reference neuron u.
        """
        # Intermediate dimension d_ff
        d_ff = ref_gate.shape[0]
        
        # Affinity matrix C[u, v] = <gate_u, gate_v> + <up_u, up_v> + <down_u, down_v>
        # C has shape [d_ff, d_ff]
        C_gate = torch.mm(ref_gate.float(), target_gate.float().t())
        C_up = torch.mm(ref_up.float(), target_up.float().t())
        C_down = torch.mm(ref_down.float().t(), target_down.float())
        
        total_affinity = C_gate + C_up + C_down # [d_ff, d_ff]
        cost_matrix = -total_affinity.cpu().numpy()
        
        # Hungarian algorithm (linear sum assignment)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        perm = torch.from_numpy(col_ind).to(ref_gate.device, dtype=torch.long)
        return perm

    @torch.no_grad()
    def permute_and_merge_block(
        self,
        block: nn.Module,
        cluster_labels: torch.Tensor,
        frequencies: torch.Tensor,
    ):
        """
        Permutes neurons of cluster members to align with the dominant expert,
        then averages weights weighted by routing frequency.
        """
        router = getattr(block, "gate", getattr(block, "router", None))
        unique_labels = torch.unique(cluster_labels).cpu().tolist()
        target_k = len(unique_labels)
        
        orig_dtype = block.experts.gate_up_proj.dtype
        dev = block.experts.gate_up_proj.device
        
        # In OLMoE fused gate_up_proj: shape [num_experts, 2 * d_ff, d_model]
        total_intermediate = block.experts.gate_up_proj.shape[1]
        d_ff = total_intermediate // 2
        
        new_gate_up = []
        new_down = []
        new_router = []
        
        freqs_fp32 = frequencies.float().to(dev)
        eps = 1e-9
        
        for c in unique_labels:
            member_indices = torch.where(cluster_labels == c)[0]
            m = len(member_indices)
            
            if m == 1:
                # Single expert, no merge needed
                idx = member_indices[0]
                new_gate_up.append(block.experts.gate_up_proj.data[idx].clone())
                new_down.append(block.experts.down_proj.data[idx].clone())
                new_router.append(router.weight.data[idx].clone())
                continue
                
            # Identify dominant expert (highest routing frequency) as reference
            m_freqs = freqs_fp32[member_indices]
            dom_pos = torch.argmax(m_freqs).item()
            ref_idx = member_indices[dom_pos]
            
            ref_gu = block.experts.gate_up_proj.data[ref_idx]
            ref_gate = ref_gu[:d_ff, :]
            ref_up = ref_gu[d_ff:, :]
            ref_down = block.experts.down_proj.data[ref_idx]
            
            total_freq = m_freqs.sum()
            if total_freq <= 0:
                weights = torch.ones_like(m_freqs) / m
            else:
                weights = m_freqs / (total_freq + eps)
                
            merged_gate = torch.zeros_like(ref_gate, dtype=torch.float32)
            merged_up = torch.zeros_like(ref_up, dtype=torch.float32)
            merged_down = torch.zeros_like(ref_down, dtype=torch.float32)
            
            # Align each member to reference expert and accumulate
            for idx_in_cluster, member_idx in enumerate(member_indices):
                w = weights[idx_in_cluster]
                
                if member_idx == ref_idx:
                    # Reference expert is already aligned (identity permutation)
                    merged_gate += w * ref_gate.float()
                    merged_up += w * ref_up.float()
                    merged_down += w * ref_down.float()
                else:
                    target_gu = block.experts.gate_up_proj.data[member_idx]
                    t_gate = target_gu[:d_ff, :]
                    t_up = target_gu[d_ff:, :]
                    t_down = block.experts.down_proj.data[member_idx]
                    
                    # Compute Hungarian matching permutation
                    perm = self.compute_bipartite_neuron_matching(
                        ref_gate, ref_up, ref_down,
                        t_gate, t_up, t_down
                    )
                    
                    # Permute target neurons: perm[u] is the target neuron aligned with ref neuron u
                    perm_gate = t_gate[perm, :]
                    perm_up = t_up[perm, :]
                    perm_down = t_down[:, perm]
                    
                    merged_gate += w * perm_gate.float()
                    merged_up += w * perm_up.float()
                    merged_down += w * perm_down.float()
                    
            # Reconstruct fused gate_up_proj
            merged_gu = torch.cat([merged_gate, merged_up], dim=0).to(orig_dtype)
            new_gate_up.append(merged_gu)
            new_down.append(merged_down.to(orig_dtype))
            
            # Router row: sum of member router rows
            merged_r = torch.zeros_like(router.weight.data[0], dtype=torch.float32)
            for idx in member_indices:
                merged_r += router.weight.data[idx].float()
            new_router.append(merged_r.to(router.weight.dtype))
            
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
    ) -> Dict[int, torch.Tensor]:
        """
        Compresses all MoE blocks in the model to target_experts using MC-SMoE.
        """
        labels_map = {}
        for layer_idx, block in enumerate(self.moe_blocks):
            freqs = frequencies_per_layer[layer_idx]
            labels = self.cluster_experts_by_router_policy(block, target_experts)
            self.permute_and_merge_block(block, labels, freqs)
            labels_map[layer_idx] = labels
            
        if hasattr(self.model, "config"):
            self.model.config.num_experts = target_experts
            self.model.config.num_local_experts = target_experts
        self.num_experts = target_experts
        return labels_map
