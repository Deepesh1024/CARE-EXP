"""
benchmark/OLMoE-1B-7B/methods/faithful_baselines/sub_moe_engine.py
==================================================================
Faithful Primary-Source Implementation of Sub-MoE (AAAI 2026).
Reference: Li et al., "Sub-MoE: Efficient Mixture-of-Expert LLMs Compression
via Subspace Expert Merging" (arXiv:2506.23266).

Core Procedures:
1. Adaptive Expert Clustering: K-means on expert output representations.
2. Experts Union Decomposition: Joint SVD on concatenated expert matrices
   to extract shared left/right singular subspaces.
3. Frequency-Based Subspace Merging: Consolidates decomposed singular components
   weighted by expert routing frequencies.
4. Physical Tensor Shrinking: Excises redundant expert slots and updates routers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans

class SubMoEEngine:
    def __init__(self, model, rank_ratio: float = 0.5):
        self.model = model
        self.rank_ratio = rank_ratio
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
    def cluster_experts_by_output(self, block, expert_outputs: torch.Tensor, num_clusters: int) -> np.ndarray:
        """
        Clusters experts based on cosine distance of mean output representations.
        expert_outputs: shape [num_experts, hidden_dim]
        """
        normed = F.normalize(expert_outputs.float(), p=2, dim=-1).cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(normed)
        return kmeans.labels_

    @torch.no_grad()
    def experts_union_decomposition_merge(
        self,
        weight_list: List[torch.Tensor],
        frequencies: List[float],
        concat_dim: int = 0
    ) -> torch.Tensor:
        """
        Executes Experts Union Decomposition via Joint SVD.
        
        Args:
            weight_list: List of parameter matrices [W_1, ..., W_m] each of shape [O, I].
            frequencies: Routing frequencies for each expert in the cluster.
            concat_dim: 0 for vertical concatenation (down_proj), 1 for horizontal (gate_up).
            
        Returns:
            Reconstructed merged tensor of shape [O, I].
        """
        orig_dtype = weight_list[0].dtype
        m = len(weight_list)
        if m == 1:
            return weight_list[0]
            
        weights_fp32 = [w.float() for w in weight_list]
        total_freq = sum(frequencies)
        if total_freq <= 0:
            weights_norm = [1.0 / m] * m
        else:
            weights_norm = [f / total_freq for f in frequencies]
            
        if concat_dim == 0:
            # Vertical concatenation: [m * O, I]
            union_mat = torch.cat(weights_fp32, dim=0) # [m * O, I]
            O, I = weight_list[0].shape
            
            # Joint SVD: union_mat = U * diag(S) * Vh
            U, S, Vh = torch.linalg.svd(union_mat, full_matrices=False)
            # U: [m * O, r], S: [r], Vh: [r, I]
            r = S.shape[0]
            max_r = max(1, int(r * self.rank_ratio))
            U = U[:, :max_r]
            S = S[:max_r]
            Vh = Vh[:max_r, :]
            
            # S is folded into Vh: V_scaled = diag(S) * Vh
            Vh_scaled = torch.diag(S) @ Vh # [max_r, I]
            
            # Combine U slices weighted by routing frequency
            U_merged = torch.zeros((O, max_r), device=union_mat.device, dtype=torch.float32)
            for idx in range(m):
                u_slice = U[idx * O : (idx + 1) * O, :]
                U_merged += weights_norm[idx] * u_slice
                
            merged = U_merged @ Vh_scaled # [O, I]
            return merged.to(orig_dtype)
        else:
            # Horizontal concatenation: [O, m * I]
            union_mat = torch.cat(weights_fp32, dim=1)
            O, I = weight_list[0].shape
            
            U, S, Vh = torch.linalg.svd(union_mat, full_matrices=False)
            r = S.shape[0]
            max_r = max(1, int(r * self.rank_ratio))
            U = U[:, :max_r]
            S = S[:max_r]
            Vh = Vh[:max_r, :]
            
            U_scaled = U @ torch.diag(S) # [O, max_r]
            
            Vh_merged = torch.zeros((max_r, I), device=union_mat.device, dtype=torch.float32)
            for idx in range(m):
                vh_slice = Vh[:, idx * I : (idx + 1) * I]
                Vh_merged += weights_norm[idx] * vh_slice
                
            merged = U_scaled @ Vh_merged # [O, I]
            return merged.to(orig_dtype)

    @torch.no_grad()
    def compress_block(
        self,
        block,
        cluster_labels: np.ndarray,
        frequencies: np.ndarray
    ):
        """
        Compresses an OlmoeSparseMoeBlock to target cluster count using Sub-MoE.
        """
        router = getattr(block, "gate", getattr(block, "router", None))
        unique_labels = np.unique(cluster_labels)
        target_k = len(unique_labels)
        
        dev = block.experts.gate_up_proj.device
        orig_dtype = block.experts.gate_up_proj.dtype
        
        new_gate_up = []
        new_down = []
        new_router = []
        
        for c in unique_labels:
            member_indices = np.where(cluster_labels == c)[0]
            c_freqs = [float(frequencies[idx]) for idx in member_indices]
            
            # 1. Decompose & merge down_proj (vertical concatenation)
            down_weights = [block.experts.down_proj.data[idx] for idx in member_indices]
            merged_down = self.experts_union_decomposition_merge(down_weights, c_freqs, concat_dim=0)
            new_down.append(merged_down)
            
            # 2. Decompose & merge gate_up_proj (horizontal concatenation)
            gu_weights = [block.experts.gate_up_proj.data[idx] for idx in member_indices]
            merged_gu = self.experts_union_decomposition_merge(gu_weights, c_freqs, concat_dim=1)
            new_gate_up.append(merged_gu)
            
            # 3. Router row: sum of member rows
            c_router = torch.zeros_like(router.weight.data[0])
            for idx in member_indices:
                c_router += router.weight.data[idx]
            new_router.append(c_router)
            
        block.experts.gate_up_proj = nn.Parameter(torch.stack(new_gate_up, dim=0))
        block.experts.down_proj = nn.Parameter(torch.stack(new_down, dim=0))
        router.weight = nn.Parameter(torch.stack(new_router, dim=0))
        block.experts.num_experts = target_k
        router.num_experts = target_k
        if hasattr(router, "out_features"):
            router.out_features = target_k
            
    def compress_model(self, target_experts: int, expert_outputs_per_layer: Dict[int, torch.Tensor], frequencies_per_layer: Dict[int, np.ndarray]):
        """Compresses all layers in the model to target_experts count."""
        for layer_idx, block in enumerate(self.moe_blocks):
            exp_out = expert_outputs_per_layer[layer_idx]
            freqs = frequencies_per_layer[layer_idx]
            labels = self.cluster_experts_by_output(block, exp_out, target_experts)
            self.compress_block(block, labels, freqs)
            
        if hasattr(self.model, "config"):
            self.model.config.num_experts = target_experts
            self.model.config.num_local_experts = target_experts
        self.num_experts = target_experts
