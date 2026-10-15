"""
care_large_moe_systems/model_reconstruction.py
Provides deterministic model loading and trajectory replay for OLMoE and Qwen3.
Computes precise parameter accounting (total, expert, active).
"""

import os
import json
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

def count_parameters(model, top_k: int = 8) -> Dict[str, Any]:
    """
    Computes exact parameter counts:
    - total_params
    - expert_params
    - non_expert_params
    - router_params
    - active_params_per_token
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    expert_params = 0
    router_params = 0
    num_moe_layers = 0
    num_experts = None
    
    for name, module in model.named_modules():
        if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
            num_moe_layers += 1
            gate_mod = getattr(module, "gate", getattr(module, "router", None))
            if gate_mod is not None:
                for p in gate_mod.parameters():
                    router_params += p.numel()
                    
            exp_mod = module.experts
            layer_exp_params = 0
            for p in exp_mod.parameters():
                layer_exp_params += p.numel()
            expert_params += layer_exp_params
            
            if hasattr(exp_mod, "num_experts"):
                num_experts = exp_mod.num_experts
            elif hasattr(gate_mod, "weight"):
                num_experts = gate_mod.weight.shape[0]

    if num_experts is None:
        num_experts = getattr(model.config, "num_experts", 64)
        
    non_expert_params = total_params - expert_params
    
    # Active parameters per token:
    # Non-expert params + all router gates + (top_k / num_experts) * expert_params
    active_expert_params = int((top_k / max(num_experts, 1)) * expert_params)
    active_params = non_expert_params + active_expert_params

    return {
        "total_parameters": int(total_params),
        "total_parameters_b": round(total_params / 1e9, 3),
        "expert_parameters": int(expert_params),
        "expert_parameters_b": round(expert_params / 1e9, 3),
        "non_expert_parameters": int(non_expert_params),
        "non_expert_parameters_b": round(non_expert_params / 1e9, 3),
        "router_parameters": int(router_params),
        "active_parameters_per_token": int(active_params),
        "active_parameters_b": round(active_params / 1e9, 3),
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "num_moe_layers": int(num_moe_layers),
        "size_bf16_gb": round((total_params * 2) / (1024**3), 2)
    }

# -------------------------------------------------------------
# OLMoE-1B-7B Deterministic Physical Merge Replay
# -------------------------------------------------------------
@torch.no_grad()
def merge_olmoe_step(moe_blocks, i: int, j: int):
    """
    Performs physical expert merge for OLMoE (OlmoeSparseMoeBlock).
    OlmoeExperts stores fused 3D tensors: gate_up_proj [E, 2*I, H], down_proj [E, H, I].
    """
    if i > j:
        i, j = j, i
        
    for block in moe_blocks:
        exp_mod = block.experts
        gate_mod = getattr(block, "gate", getattr(block, "router", None))
        
        dev = exp_mod.gate_up_proj.device
        n = exp_mod.num_experts
        
        # 1. Average expert i and j
        avg_gate_up = (exp_mod.gate_up_proj.data[i] + exp_mod.gate_up_proj.data[j]) / 2.0
        avg_down = (exp_mod.down_proj.data[i] + exp_mod.down_proj.data[j]) / 2.0
        avg_router = (gate_mod.weight.data[i] + gate_mod.weight.data[j]) / 2.0
        
        # 2. Slice out j
        keep_idx = torch.tensor([k for k in range(n) if k != j], device=dev)
        new_gate_up = exp_mod.gate_up_proj.data[keep_idx]
        new_down = exp_mod.down_proj.data[keep_idx]
        new_router = gate_mod.weight.data[keep_idx]
        
        # 3. Update slot i
        new_gate_up[i] = avg_gate_up
        new_down[i] = avg_down
        new_router[i] = avg_router
        
        exp_mod.gate_up_proj = nn.Parameter(new_gate_up)
        exp_mod.down_proj = nn.Parameter(new_down)
        gate_mod.weight = nn.Parameter(new_router)
        exp_mod.num_experts = n - 1
        if hasattr(gate_mod, "num_experts"):
            gate_mod.num_experts = n - 1

def reconstruct_olmoe_model(
    model,
    target_experts: int,
    trajectory_path: str
) -> Tuple[Any, float]:
    """
    Replays recorded physical merges on pristine OLMoE model to reach target_experts.
    """
    moe_blocks = [l.mlp for l in model.model.layers]
    current_experts = moe_blocks[0].experts.num_experts
    
    if current_experts == target_experts:
        return model, 0.0
        
    merges_needed = current_experts - target_experts
    if merges_needed < 0:
        raise ValueError(f"Cannot expand experts from {current_experts} to {target_experts}")
        
    with open(trajectory_path, "r") as f:
        traj_data = json.load(f)
        
    trace = traj_data["trace"]
    t0 = time.perf_counter()
    
    applied = 0
    for step in trace:
        if applied >= merges_needed:
            break
        selected_pair = step.get("selected_pair")
        if selected_pair is not None:
            i, j = selected_pair
            merge_olmoe_step(moe_blocks, i, j)
            applied += 1
            
    replay_time = time.perf_counter() - t0
    
    model.config.num_experts = target_experts
    for block in moe_blocks:
        block.experts.num_experts = target_experts
        if hasattr(block.gate, "num_experts"):
            block.gate.num_experts = target_experts
            
    print(f"[OLMoE Replay] Applied {applied} merges in {replay_time:.3f}s. Target: {target_experts} experts.")
    return model, replay_time

# -------------------------------------------------------------
# Qwen3-30B-A3B Deterministic Physical Merge Replay
# -------------------------------------------------------------
@torch.no_grad()
def merge_qwen3_step(moe_blocks, i: int, j: int):
    """
    Performs physical expert merge for Qwen3-30B-A3B (Qwen3MoeSparseMoeBlock).
    """
    if i > j:
        i, j = j, i
    for block in moe_blocks:
        dev = block.experts.gate_up_proj.device
        n = block.experts.num_experts
        
        avg_gate_up = (block.experts.gate_up_proj.data[i] + block.experts.gate_up_proj.data[j]) / 2.0
        avg_down = (block.experts.down_proj.data[i] + block.experts.down_proj.data[j]) / 2.0
        avg_router = (block.gate.weight.data[i] + block.gate.weight.data[j]) / 2.0
        
        keep_idx = torch.tensor([k for k in range(n) if k != j], device=dev)
        new_gate_up = block.experts.gate_up_proj.data[keep_idx]
        new_down = block.experts.down_proj.data[keep_idx]
        new_router = block.gate.weight.data[keep_idx]
        
        new_gate_up[i] = avg_gate_up
        new_down[i] = avg_down
        new_router[i] = avg_router
        
        block.experts.gate_up_proj = nn.Parameter(new_gate_up)
        block.experts.down_proj = nn.Parameter(new_down)
        block.gate.weight = nn.Parameter(new_router)
        block.experts.num_experts = n - 1
        block.gate.num_experts = n - 1

def reconstruct_qwen3_model(
    model,
    target_experts: int,
    results_path: str
) -> Tuple[Any, float]:
    """
    Replays recorded physical merges on Qwen3-30B-A3B to reach target_experts.
    """
    moe_blocks = [l.mlp for l in model.model.layers]
    current_experts = moe_blocks[0].experts.num_experts
    
    if current_experts == target_experts:
        return model, 0.0
        
    merges_needed = current_experts - target_experts
    with open(results_path, "r") as f:
        data = json.load(f)
        
    trace = data["trace"][:merges_needed]
    t0 = time.perf_counter()
    
    for idx, step in enumerate(trace):
        i, j = step["selected_pair"]
        merge_qwen3_step(moe_blocks, i, j)
        
    replay_time = time.perf_counter() - t0
    model.config.num_experts = target_experts
    if hasattr(model.config, "num_local_experts"):
        model.config.num_local_experts = target_experts
        
    print(f"[Qwen3 Replay] Applied {len(trace)} merges in {replay_time:.3f}s -> {target_experts} experts.")
    return model, replay_time
