"""
audit_repair_pass/full_64e_validation/care_capability_v23.py
============================================================
Versioned, Audited CARE-COM Capability Extraction Engine (v2.3).

Forensic Fixes Applied over v2.1:
1. Dual-Architecture Dispatch: Supports both fused 3D tensors (OlmoeExperts)
   and unfused nn.ModuleList (OlmoeMLP/MockExpert).
2. Tensor Dimension Alignment: In fused OlmoeExperts, forward projection for
   expert `e` uses `F.linear(x, gate_up_proj[e])` and `F.linear(act, down_proj[e])`
   matching HF parameter layouts.
3. Correct Layer-Normalization Indexing: Fixes the critical bug where
   `layer_capabilities[:, layer_idx, :]` sliced the expert dimension instead
   of the layer dimension, which inadvertently restricted normalization to
   the first 16 experts. Slices `layer_capabilities[layer_idx, :, :]`.
4. Dynamic Active Expert Tracking: Profiles exactly `engine.current_num_experts`
   at each step, ensuring zero stale or out-of-bounds indices.
5. Full-Spectrum Nomination: Generates candidate pairs across all N*(N-1)/2
   possible pairs in the active pool [0, N_t - 1].
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

def compute_global_capability_vectors_v23(
    model: torch.nn.Module,
    moe_blocks: list,
    current_num_experts: int,
    df_tokens,
    samples_per_axis: int = 20,
    device: str = "cuda"
) -> np.ndarray:
    """
    Computes global capability vectors across all layers for all active experts.
    
    Args:
        model: Active causal language model.
        moe_blocks: List of MoE blocks (e.g. OlmoeSparseMoeBlock).
        current_num_experts: Exact count of active experts in the current state.
        df_tokens: DataFrame containing token sequences with 'axis_idx', 'input_ids', 'attention_mask'.
        samples_per_axis: Number of sequence samples per capability axis.
        device: Target execution device.
        
    Returns:
        np.ndarray of shape [current_num_experts, num_layers * 10]
    """
    model.eval()
    num_layers = len(moe_blocks)
    num_experts = current_num_experts
    
    # layer_capabilities: [num_layers, num_experts, 10]
    layer_capabilities = np.zeros((num_layers, num_experts, 10), dtype=np.float32)
    
    # Determine whether blocks use fused 3D tensors or ModuleList
    first_block = moe_blocks[0]
    is_fused = hasattr(first_block.experts, "gate_up_proj")
    
    for axis_idx in range(10):
        axis_tokens = df_tokens[df_tokens["axis_idx"] == axis_idx].head(samples_per_axis)
        if len(axis_tokens) == 0:
            continue
            
        for _, row in axis_tokens.iterrows():
            input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(device)
            mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(device)
            valid_len = mask.sum().item()
            if valid_len == 0:
                continue
            mask_t = mask.unsqueeze(-1)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                hidden_states = out.hidden_states
                
                for layer_idx, block in enumerate(moe_blocks):
                    x = hidden_states[layer_idx] # Input to layer
                    
                    if is_fused:
                        # OlmoeExperts fused 3D tensor:
                        # gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
                        # down_proj: [num_experts, hidden_size, intermediate_size]
                        for exp_idx in range(num_experts):
                            gate_up = F.linear(x, block.experts.gate_up_proj[exp_idx])
                            gate, up = gate_up.chunk(2, dim=-1)
                            act = F.silu(gate) * up
                            out_e = F.linear(act, block.experts.down_proj[exp_idx])
                            out_e = out_e * mask_t
                            norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                            layer_capabilities[layer_idx, exp_idx, axis_idx] += norm_val
                    else:
                        # Unfused ModuleList
                        for exp_idx in range(num_experts):
                            exp = block.experts[exp_idx]
                            gate = F.silu(F.linear(x, exp.gate_proj.weight))
                            up = F.linear(x, exp.up_proj.weight)
                            out_e = F.linear(gate * up, exp.down_proj.weight)
                            out_e = out_e * mask_t
                            norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                            layer_capabilities[layer_idx, exp_idx, axis_idx] += norm_val
                            
        # Average over number of samples for this axis with non-zero guard
        if len(axis_tokens) > 0:
            layer_capabilities[:, :, axis_idx] /= len(axis_tokens)
        
    # Layer-normalization: normalize each layer's [num_experts, 10] slice independently
    # AUDIT FIX: slice layer_capabilities[layer_idx, :, :] (NOT [:, layer_idx, :])
    for layer_idx in range(num_layers):
        layer_C = layer_capabilities[layer_idx, :, :] # Shape: [num_experts, 10]
        norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        layer_capabilities[layer_idx, :, :] = layer_C / norms
        
    # Concatenate across layers: transpose to [num_experts, num_layers, 10] -> reshape [num_experts, num_layers * 10]
    C_concat = layer_capabilities.transpose(1, 0, 2).reshape(num_experts, num_layers * 10)
    
    # Global unit-normalization across the concatenated vector
    global_norms = np.linalg.norm(C_concat, axis=1, keepdims=True)
    global_norms[global_norms == 0] = 1.0
    C_final = C_concat / global_norms
    
    # AUDIT DEFENSE: Assert capabilities are non-trivial and not all zeros
    assert np.any(C_final != 0), "Capability extraction produced all-zero vectors! Check architecture forward hook / dispatch."
    assert not np.isnan(C_final).any(), "NaN values detected in capability matrix!"
    
    return C_final

def pairwise_capability_distances_v23(C: np.ndarray) -> Dict[int, Dict[int, float]]:
    """
    Computes pairwise L2 Euclidean distances between all active experts in capability space.
    
    Args:
        C: Array of shape [num_experts, feature_dim]
        
    Returns:
        Nested dict distances[i][j] = float
    """
    num_experts = C.shape[0]
    distances = {i: {} for i in range(num_experts)}
    all_d = []
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            d = float(np.linalg.norm(C[i] - C[j]))
            distances[i][j] = d
            distances[j][i] = d
            all_d.append(d)
            
    # AUDIT DEFENSE: Verify pairwise distances have non-zero variance
    if len(all_d) > 1:
        dist_std = float(np.std(all_d))
        assert dist_std > 1e-6, f"Pairwise distances have zero or degenerate variance (std={dist_std:.8f})! All pairs look identical."
        
    return distances

def generate_candidate_pool_v23(distances: Dict[int, Dict[int, float]], pool_size: int) -> List[Tuple[int, int]]:
    """
    Generates unique sorted (i, j) candidate pairs based on smallest capability distance.
    Guarantees 0 <= i < j < num_experts.
    """
    pairs = []
    seen = set()
    num_experts = len(distances)
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            if j in distances[i]:
                pair = (i, j)
                if pair not in seen:
                    pairs.append((*pair, distances[i][j]))
                    seen.add(pair)
                    
    pairs.sort(key=lambda x: x[2])
    candidates = [(x[0], x[1]) for x in pairs[:pool_size]]
    
    # AUDIT DEFENSE: Assert candidate pool is non-empty and correctly bounded
    assert len(candidates) > 0, "Candidate pool is empty!"
    for i, j in candidates:
        assert 0 <= i < j < num_experts, f"Candidate pair ({i}, {j}) out of active bounds [0, {num_experts})!"
        
    return candidates
