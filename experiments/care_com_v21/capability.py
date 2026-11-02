"""
experiments/care_com_v21/capability.py
============================================================
Audited and Corrected CARE-COM Capability Extraction Engine.

Forensic Fixes Applied:
1. Dual-Architecture Dispatch: Supports both fused 3D tensor expert modules
   (e.g., OlmoeExperts with gate_up_proj/down_proj) and unfused nn.ModuleList
   of expert blocks (gate_proj/up_proj/down_proj or w1/w3/w2).
2. Dynamic Active Expert Identification: Correctly extracts the exact number
   of active experts from the router weight matrix or explicit active_num_experts,
   preventing stale indexing or indexing out-of-range experts.
3. Layer-Normalization Slicing Correction: Slices `layer_capabilities[layer_idx, :, :]`
   (axis 0 = layer, axis 1 = experts, axis 2 = 10 capability axes).
   Fixes historical bug where `layer_capabilities[:, layer_idx, :]` sliced the expert
   dimension, accidentally restricting candidate generation and normalization to
   the first 16 experts.
4. Numerical Stability & Zero-Vector Protection: Explicit division-by-zero guards
   and NaN/Inf sanitization for zero or dead experts.
5. Explicit Failure Mode: Fails immediately with descriptive errors if an
   unsupported architecture is passed, rather than silently returning a truncated
   or corrupt expert set.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

def get_active_expert_count(moe_block) -> int:
    """Extracts the exact number of active experts for a given MoE block."""
    router = getattr(moe_block, "gate", getattr(moe_block, "router", None))
    if router is not None and hasattr(router, "weight"):
        return router.weight.shape[0]
    experts = getattr(moe_block, "experts", None)
    if experts is not None:
        if hasattr(experts, "num_experts"):
            return experts.num_experts
        if hasattr(experts, "gate_up_proj"):
            return experts.gate_up_proj.shape[0]
        if isinstance(experts, (list, torch.nn.ModuleList)):
            return len(experts)
    raise ValueError(f"Unable to determine expert count for MoE block of type {type(moe_block)}")

def compute_global_capability_vectors(
    model: torch.nn.Module,
    moe_blocks: list,
    df_tokens,
    device: str = "cuda:0",
    current_num_experts: Optional[int] = None,
    samples_per_axis: int = 20
) -> np.ndarray:
    """
    Computes normalized global capability vectors across all layers for active experts.
    Returns: numpy array of shape [num_experts, num_layers * 10]
    where each layer's capability (10 dimensions) is computed identically and layer-normalized.
    """
    if len(moe_blocks) == 0:
        raise ValueError("moe_blocks list cannot be empty.")

    model.eval()
    num_layers = len(moe_blocks)
    first_block = moe_blocks[0]

    # Determine exact count of active experts
    if current_num_experts is not None:
        num_experts = current_num_experts
    else:
        num_experts = get_active_expert_count(first_block)

    if num_experts <= 0:
        raise ValueError(f"Invalid active expert count: {num_experts}")

    # Determine whether blocks use fused 3D tensors or ModuleList
    is_fused = hasattr(first_block.experts, "gate_up_proj")

    print(f"Computing Capability Vectors for {num_experts} active experts across {num_layers} layers (fused={is_fused})...")
    
    # layer_capabilities: [num_layers, num_experts, 10]
    layer_capabilities = np.zeros((num_layers, num_experts, 10), dtype=np.float32)

    # Normalize probe inputs into a list of (input_ids, attention_mask) per axis
    axis_samples: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {a: [] for a in range(10)}
    
    if hasattr(df_tokens, "iterrows"):
        # Pandas DataFrame
        for axis_idx in range(10):
            matched = df_tokens[df_tokens["axis_idx"] == axis_idx].head(samples_per_axis)
            for _, row in matched.iterrows():
                inp = torch.tensor(row['input_ids']).unsqueeze(0) if not isinstance(row['input_ids'], torch.Tensor) else row['input_ids']
                if inp.ndim == 1:
                    inp = inp.unsqueeze(0)
                msk = torch.tensor(row['attention_mask']).unsqueeze(0) if not isinstance(row['attention_mask'], torch.Tensor) else row['attention_mask']
                if msk.ndim == 1:
                    msk = msk.unsqueeze(0)
                axis_samples[axis_idx].append((inp, msk))
    elif isinstance(df_tokens, (list, tuple)):
        # List of batches
        for idx, item in enumerate(df_tokens):
            axis_idx = idx % 10
            if len(axis_samples[axis_idx]) >= samples_per_axis:
                continue
            if isinstance(item, dict):
                inp = item.get("input_ids")
                msk = item.get("attention_mask")
                if inp is not None:
                    if not isinstance(inp, torch.Tensor):
                        inp = torch.tensor(inp)
                    if inp.ndim == 1:
                        inp = inp.unsqueeze(0)
                    if msk is None:
                        msk = torch.ones_like(inp)
                    elif not isinstance(msk, torch.Tensor):
                        msk = torch.tensor(msk)
                    if msk.ndim == 1:
                        msk = msk.unsqueeze(0)
                    axis_samples[axis_idx].append((inp, msk))
            elif isinstance(item, torch.Tensor):
                inp = item if item.ndim == 2 else item.unsqueeze(0)
                msk = torch.ones_like(inp)
                axis_samples[axis_idx].append((inp, msk))
        # If some axes are empty, replicate available samples across all 10 axes
        available = [s for samples in axis_samples.values() for s in samples]
        if available:
            for a in range(10):
                if not axis_samples[a]:
                    axis_samples[a] = list(available[:samples_per_axis])
    elif isinstance(df_tokens, torch.Tensor):
        inp = df_tokens if df_tokens.ndim == 2 else df_tokens.unsqueeze(0)
        msk = torch.ones_like(inp)
        for a in range(10):
            axis_samples[a] = [(inp, msk)]

    for axis_idx in tqdm(range(10), desc="Capability Axes", leave=False):
        samples = axis_samples[axis_idx]
        if len(samples) == 0:
            continue

        for inp, msk in samples:
            input_ids = inp.to(device)
            mask = msk.to(device)
            valid_len = mask.sum().item()
            if valid_len == 0:
                continue
            mask_t = mask.unsqueeze(-1)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                hidden_states = out.hidden_states

                for layer_idx, block in enumerate(moe_blocks):
                    # Hidden states start with embeddings; index layer_idx is input to layer block
                    x = hidden_states[layer_idx]

                    if is_fused:
                        # Fused 3D tensor expert representation (e.g., OLMoE)
                        # gate_up_proj: [num_experts, 2 * intermediate, hidden]
                        # down_proj: [num_experts, hidden, intermediate]
                        for exp_idx in range(num_experts):
                            gate_up = F.linear(x, block.experts.gate_up_proj[exp_idx])
                            gate, up = gate_up.chunk(2, dim=-1)
                            act = F.silu(gate) * up
                            out_e = F.linear(act, block.experts.down_proj[exp_idx])
                            out_e = out_e * mask_t
                            norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum() / valid_len).item()
                            layer_capabilities[layer_idx, exp_idx, axis_idx] += norm_val
                    else:
                        # Unfused ModuleList of expert modules
                        for exp_idx in range(num_experts):
                            exp = block.experts[exp_idx]
                            if hasattr(exp, "gate_proj") and hasattr(exp, "up_proj") and hasattr(exp, "down_proj"):
                                gate = F.silu(F.linear(x, exp.gate_proj.weight))
                                up = F.linear(x, exp.up_proj.weight)
                                out_e = F.linear(gate * up, exp.down_proj.weight)
                            elif hasattr(exp, "w1") and hasattr(exp, "w3") and hasattr(exp, "w2"):
                                gate = F.silu(F.linear(x, exp.w1.weight))
                                up = F.linear(x, exp.w3.weight)
                                out_e = F.linear(gate * up, exp.w2.weight)
                            else:
                                raise TypeError(f"Unsupported expert module type: {type(exp)}")
                            out_e = out_e * mask_t
                            norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum() / valid_len).item()
                            layer_capabilities[layer_idx, exp_idx, axis_idx] += norm_val

        # Average over number of evaluated samples for this axis
        layer_capabilities[:, :, axis_idx] /= len(samples)

    # Layer-normalization: normalize each layer's [num_experts, 10] slice independently
    # AUDIT FIX: slice layer_capabilities[layer_idx, :, :] (NOT [:, layer_idx, :])
    for layer_idx in range(num_layers):
        layer_C = layer_capabilities[layer_idx, :, :]  # Shape: [num_experts, 10]
        norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        layer_capabilities[layer_idx, :, :] = layer_C / norms

    # Sanitize non-finite values if any
    layer_capabilities = np.nan_to_num(layer_capabilities, nan=0.0, posinf=1.0, neginf=-1.0)

    # Concatenate across layers: transpose to [num_experts, num_layers, 10] -> reshape [num_experts, num_layers * 10]
    C_concat = layer_capabilities.transpose(1, 0, 2).reshape(num_experts, num_layers * 10)

    # Global unit-normalization across concatenated vector
    global_norms = np.linalg.norm(C_concat, axis=1, keepdims=True)
    global_norms[global_norms == 0] = 1.0
    C_final = C_concat / global_norms
    C_final = np.nan_to_num(C_final, nan=0.0, posinf=1.0, neginf=-1.0)

    return C_final

def pairwise_capability_distances(C: np.ndarray) -> Dict[int, Dict[int, float]]:
    """
    Computes pairwise L2 Euclidean distances between all active experts in capability space.
    Returns: distances[i][j] = float
    """
    num_experts = C.shape[0]
    distances = {i: {} for i in range(num_experts)}
    for i in range(num_experts):
        for j in range(num_experts):
            if i != j:
                distances[i][j] = float(np.linalg.norm(C[i] - C[j]))
    return distances
