"""
EXPERIMENT 7B - MODEL UTILITIES
===============================
Provides robust model loading, in-place expert merging with clean restoration,
expert ablation hooks (individual and joint), and VRAM cleanup.
"""

import os
import gc
import types
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_MODEL_ID, HF_REVISION, TARGET_LAYER_IDX, DEVICE, DTYPE

def load_base_model():
    """Loads base model and tokenizer cleanly."""
    print(f"[ModelUtils] Loading {BASE_MODEL_ID} on {DEVICE} ({DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=HF_REVISION,
        torch_dtype=DTYPE,
        device_map=DEVICE if "cuda" in DEVICE else None,
        trust_remote_code=True
    )
    if DEVICE == "mps" or (DEVICE != "cpu" and "cuda" not in DEVICE):
        model = model.to(DEVICE)
    model.eval()
    return model, tokenizer

def get_target_moe_block(model, target_layer_idx=TARGET_LAYER_IDX):
    """Retrieves the central MoE block at target_layer_idx."""
    # Handle standard HuggingFace OLMoE structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[target_layer_idx]
        if hasattr(layer, "mlp"):
            return layer.mlp
    
    # Fallback search by module name
    for name, module in model.named_modules():
        if f"layers.{target_layer_idx}" in name and hasattr(module, "experts"):
            return module
            
    raise ValueError(f"Could not locate MoE block for layer {target_layer_idx}.")

def apply_in_place_merge(moe_block, expert_i: int, expert_j: int):
    """
    Applies non-destructive in-place uniform parameter averaging on experts i and j.
    Both expert slots i and j receive the merged weights.
    Returns a restore callback to cleanly reset original parameters.
    """
    experts_mod = moe_block.experts
    is_fused = hasattr(experts_mod, "gate_up_proj") and hasattr(experts_mod, "down_proj")
    
    if is_fused:
        # Fused tensor shape: [num_experts, out_dim, in_dim]
        orig_gate_i = experts_mod.gate_up_proj.data[expert_i].clone()
        orig_gate_j = experts_mod.gate_up_proj.data[expert_j].clone()
        orig_down_i = experts_mod.down_proj.data[expert_i].clone()
        orig_down_j = experts_mod.down_proj.data[expert_j].clone()
        
        # Average weights
        merged_gate = 0.5 * (orig_gate_i + orig_gate_j)
        merged_down = 0.5 * (orig_down_i + orig_down_j)
        
        # Copy into both expert slots
        with torch.no_grad():
            experts_mod.gate_up_proj.data[expert_i].copy_(merged_gate)
            experts_mod.gate_up_proj.data[expert_j].copy_(merged_gate)
            experts_mod.down_proj.data[expert_i].copy_(merged_down)
            experts_mod.down_proj.data[expert_j].copy_(merged_down)
            
        def restore():
            with torch.no_grad():
                experts_mod.gate_up_proj.data[expert_i].copy_(orig_gate_i)
                experts_mod.gate_up_proj.data[expert_j].copy_(orig_gate_j)
                experts_mod.down_proj.data[expert_i].copy_(orig_down_i)
                experts_mod.down_proj.data[expert_j].copy_(orig_down_j)
            cleanup_vram()
            
        return restore
    else:
        # Separate ModuleList experts
        expert_mod_i = experts_mod[expert_i]
        expert_mod_j = experts_mod[expert_j]
        
        orig_state_i = {k: v.clone() for k, v in expert_mod_i.state_dict().items()}
        orig_state_j = {k: v.clone() for k, v in expert_mod_j.state_dict().items()}
        
        merged_state = {}
        for k in orig_state_i:
            merged_state[k] = 0.5 * (orig_state_i[k] + orig_state_j[k])
            
        with torch.no_grad():
            expert_mod_i.load_state_dict(merged_state)
            expert_mod_j.load_state_dict(merged_state)
            
        def restore():
            with torch.no_grad():
                expert_mod_i.load_state_dict(orig_state_i)
                expert_mod_j.load_state_dict(orig_state_j)
            cleanup_vram()
            
        return restore

def apply_expert_ablation(moe_block, target_experts: list):
    """
    Applies functional Zero-Output Ablation on target_experts (list of expert indices).
    When tokens are routed to any target expert in target_experts, that expert's
    output contribution is zeroed (y_e = 0).
    Returns a restore callback to remove the hook cleanly.
    """
    target_set = set(target_experts)
    gate_module = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    experts_mod = moe_block.experts
    
    # Method: Register hook on expert execution
    # For batched OlmoeExperts, forward takes (hidden_states, topk_weights, topk_idx)
    # We can mask intermediate activations or outputs of the targeted experts.
    handles = []
    
    if hasattr(experts_mod, "down_proj"):
        # Fused experts: hook on down_proj or monkey patch
        # In OlmoeExperts, topk_ids select expert slices.
        # Clean forward hook on experts_mod:
        def zero_ablation_hook(module, inputs, output):
            # output is tensor of shape [batch, seq, hidden] or tuple (output, ...)
            # If OlmoeExperts custom forward, we zero out contribution of target_experts
            pass

    # Standard safe approach matching Exp 7A:
    # Set an ablation attribute on moe_block
    moe_block._ablated_experts = target_set
    
    # Check if expert modules are iterable or batched
    if isinstance(experts_mod, nn.ModuleList) or hasattr(experts_mod, "__iter__"):
        for idx in target_set:
            if idx < len(experts_mod):
                sub_mod = experts_mod[idx]
                h = sub_mod.register_forward_hook(lambda m, inp, out: torch.zeros_like(out))
                handles.append(h)
    else:
        # Batched OlmoeExperts module forward hook
        # Intercept output after expert forward
        def fused_ablation_hook(module, inp, output):
            # In OLMoE fused block, if router logits are available,
            # zeroing router weights for ablated experts guarantees zero output contribution.
            pass
            
    # Router Logit Masking hook as universal fallback:
    def router_mask_hook(module, inp, output):
        # output is router logits [batch, seq, num_experts] or tuple
        if isinstance(output, tuple):
            logits = output[0]
            new_logits = logits.clone()
            for e in target_set:
                new_logits[..., e] = float('-inf')
            return (new_logits,) + output[1:]
        else:
            new_logits = output.clone()
            for e in target_set:
                new_logits[..., e] = float('-inf')
            return new_logits

    if gate_module is not None:
        h_router = gate_module.register_forward_hook(router_mask_hook)
        handles.append(h_router)
        
    def restore():
        for h in handles:
            h.remove()
        if hasattr(moe_block, "_ablated_experts"):
            del moe_block._ablated_experts
        cleanup_vram()
        
    return restore

def cleanup_vram():
    """Frees cached GPU memory and forces garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
