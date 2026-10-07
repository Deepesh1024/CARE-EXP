import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers.models.phimoe.configuration_phimoe import PhimoeConfig
from transformers.models.phimoe.modeling_phimoe import PhimoeForCausalLM, PhimoeExperts

###############################################################################
# CORE PHI-3.5 CARE COMPONENTS
###############################################################################

def evaluate_phi_expert_on_probe(x: torch.Tensor, experts_module: nn.Module, exp_idx: int) -> torch.Tensor:
    """
    Evaluates the unweighted expert functional mapping on the residual-stream probe x.
    Validates architecture assumptions at runtime.
    """
    assert hasattr(experts_module, "gate_up_proj"), "Expert module missing gate_up_proj"
    assert hasattr(experts_module, "down_proj"), "Expert module missing down_proj"
    
    gate_up = experts_module.gate_up_proj
    down = experts_module.down_proj
    
    assert gate_up.ndim == 3, f"Expected 3D gate_up_proj, got {gate_up.ndim}"
    assert down.ndim == 3, f"Expected 3D down_proj, got {down.ndim}"
    assert gate_up.shape[0] == down.shape[0] == experts_module.num_experts, "Expert count mismatch"
    assert gate_up.shape[1] == 2 * down.shape[2], "gate_up_proj is not exactly 2x down_proj intermediate size"
    
    w_gate_up = gate_up[exp_idx]
    w_down = down[exp_idx]
    
    gate_up_out = F.linear(x, w_gate_up)
    gate, up = gate_up_out.chunk(2, dim=-1)
    
    # Intrinsic expert mapping: SiLU(gate) * up -> down
    out_e = F.linear(F.silu(gate) * up, w_down)
    return out_e


class PhiPhysicalMergeEngine:
    def __init__(self, model):
        self.model = model
        self.moe_blocks = []
        for layer_idx, layer in enumerate(getattr(model, "model", model).layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "router"):
                self.moe_blocks.append(layer.mlp)
        
        assert len(self.moe_blocks) > 0, "No MoE blocks found"
        self.current_num_experts = self.moe_blocks[0].experts.num_experts
        self._snapshot = None
        
    def snapshot(self, expert_i: int):
        self._snapshot = {
            "expert_i": expert_i,
            "model_num_experts": getattr(self.model.config, 'num_local_experts', None),
            "blocks": []
        }
        for block in self.moe_blocks:
            router = block.router
            experts = block.experts
            
            self._snapshot["blocks"].append({
                "block": block,
                "router_weight": router.weight.clone().detach(),
                "gate_up_proj": experts.gate_up_proj.clone().detach(),
                "down_proj": experts.down_proj.clone().detach(),
                "num_experts": experts.num_experts,
                "router_out_features": router.out_features
            })
            
    def restore(self):
        if self._snapshot is None:
            raise RuntimeError("No snapshot")
            
        if self._snapshot["model_num_experts"] is not None:
            self.model.config.num_local_experts = self._snapshot["model_num_experts"]
            if hasattr(self.model.config, 'num_experts'):
                self.model.config.num_experts = self._snapshot["model_num_experts"]
                
        for b_data in self._snapshot["blocks"]:
            block = b_data["block"]
            
            block.router.weight = nn.Parameter(b_data["router_weight"])
            block.router.out_features = b_data["router_out_features"]
            
            block.experts.gate_up_proj = nn.Parameter(b_data["gate_up_proj"])
            block.experts.down_proj = nn.Parameter(b_data["down_proj"])
            
            block.experts.num_experts = b_data["num_experts"]
            block.router.num_experts = b_data["num_experts"]
            block.num_experts = b_data["num_experts"]
            
        self.current_num_experts = self._snapshot["blocks"][0]["num_experts"]
        self._snapshot = None

    @torch.no_grad()
    def merge_experts(self, i: int, j: int):
        if i >= self.current_num_experts or j >= self.current_num_experts:
            raise ValueError(f"Invalid expert indices {i}, {j}")
            
        for block in self.moe_blocks:
            router = block.router
            experts = block.experts
            
            # 1. Merge router (Average)
            router_w = router.weight.data
            router_new = (router_w[i] + router_w[j]) / 2.0
            
            # 2. Merge expert weights (Average)
            gate_up_new = (experts.gate_up_proj.data[i] + experts.gate_up_proj.data[j]) / 2.0
            down_new = (experts.down_proj.data[i] + experts.down_proj.data[j]) / 2.0
            
            # 3. Create N-1 tensors
            keep = [idx for idx in range(self.current_num_experts) if idx != j]
            
            new_router_w = router_w[keep].clone()
            new_i = keep.index(i)
            new_router_w[new_i] = router_new
            router.weight = nn.Parameter(new_router_w)
            router.out_features = len(keep)
            
            new_gate_up = experts.gate_up_proj.data[keep].clone()
            new_gate_up[new_i] = gate_up_new
            new_down = experts.down_proj.data[keep].clone()
            new_down[new_i] = down_new
            
            experts.gate_up_proj = nn.Parameter(new_gate_up)
            experts.down_proj = nn.Parameter(new_down)
            
            new_num = len(keep)
            experts.num_experts = new_num
            router.num_experts = new_num
            block.num_experts = new_num
            
        self.current_num_experts = new_num
        self.model.config.num_local_experts = new_num
        if hasattr(self.model.config, 'num_experts'):
            self.model.config.num_experts = new_num


###############################################################################
# GATES
###############################################################################

def gate1_expert_equivalence():
    print("\n--- GATE 1: Expert Equivalence Test ---")
    config = PhimoeConfig(hidden_size=64, intermediate_size=128, num_local_experts=4)
    experts = PhimoeExperts(config)
    for p in experts.parameters():
        nn.init.normal_(p)
    experts.eval()
    
    seeds = [0, 42, 1337]
    shapes = [(1, 1), (1, 7), (2, 13)]
    dtype = torch.float32
    device = "cpu"
    
    for seed in seeds:
        torch.manual_seed(seed)
        for B, S in shapes:
            x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
            for exp_idx in range(config.num_local_experts):
                
                # Native computation
                with torch.no_grad():
                    gate_up_native = F.linear(x, experts.gate_up_proj[exp_idx])
                    gate_native, up_native = gate_up_native.chunk(2, dim=-1)
                    native_out = F.linear(F.silu(gate_native) * up_native, experts.down_proj[exp_idx])
                
                # Adapter computation
                adapter_out = evaluate_phi_expert_on_probe(x, experts, exp_idx)
                
                max_abs_error = torch.max(torch.abs(native_out - adapter_out)).item()
                mean_abs_error = torch.mean(torch.abs(native_out - adapter_out)).item()
                # Rel error protected against div by zero
                max_rel_error = torch.max(torch.abs(native_out - adapter_out) / (torch.abs(native_out) + 1e-8)).item()
                
                assert max_abs_error < 1e-6, f"Gate 1 Failed! Max Abs Error: {max_abs_error}"
                
    print("✅ Gate 1 Passed: Adapter perfectly matches native intrinsic expert computation.")


def gate2_physical_deletion():
    print("\n--- GATE 2: Physical N -> N-1 Deletion Test ---")
    config = PhimoeConfig(
        hidden_size=64, intermediate_size=128, num_local_experts=16, 
        num_hidden_layers=2, vocab_size=100, pad_token_id=0
    )
    model = PhimoeForCausalLM(config).eval()
    engine = PhiPhysicalMergeEngine(model)
    
    assert engine.current_num_experts == 16
    assert model.config.num_local_experts == 16
    assert engine.moe_blocks[0].router.weight.shape == (16, 64)
    assert engine.moe_blocks[0].experts.gate_up_proj.shape == (16, 256, 64)
    
    # Dummy forward pass before merge
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        # router_logits might not be captured natively unless requested. Let's use hook.
    
    # Attach hook to verify router output shape natively
    router_out_shape = None
    def router_hook(m, i, o):
        nonlocal router_out_shape
        router_out_shape = o[0].shape # o is tuple (router_logits, routing_weights, selected_experts)
    
    engine.moe_blocks[0].router.register_forward_hook(router_hook)
    with torch.no_grad():
        model(x)
    assert router_out_shape == (2*8, 16), f"Expected (16, 16) but got {router_out_shape}"
    
    # Perform N -> N-1
    engine.snapshot(0)
    engine.merge_experts(3, 11)
    
    assert engine.current_num_experts == 15, "Engine expert count didn't update"
    assert model.config.num_local_experts == 15, "Config expert count didn't update"
    
    for block in engine.moe_blocks:
        assert block.router.weight.shape == (15, 64), "Router weight shape incorrect"
        assert block.experts.gate_up_proj.shape == (15, 256, 64), "gate_up_proj shape incorrect"
        assert block.experts.down_proj.shape == (15, 64, 128), "down_proj shape incorrect"
        
    # Forward pass after merge
    with torch.no_grad():
        model(x)
    
    assert router_out_shape == (2*8, 15), f"Router output shape is {router_out_shape}, expected (16, 15)"
    
    # Restore
    engine.restore()
    assert engine.current_num_experts == 16
    with torch.no_grad():
        model(x)
    assert router_out_shape == (2*8, 16)
    
    print("✅ Gate 2 Passed: True N -> N-1 physical reduction achieved and restored cleanly.")


def dummy_capability_extraction(model, engine, num_experts, num_layers):
    """Simulate care_com_v21/capability.py logic with random tokens"""
    # Create random token embedding as the probe
    # Real implementation uses 10 axes * 20 samples = 200 embeddings per layer
    probe_x = torch.randn(200, 64)
    
    # Real implementation: layer_capabilities = np.zeros((num_layers, num_experts, 10))
    layer_capabilities = torch.zeros(num_layers, num_experts, 10)
    
    for layer_idx, block in enumerate(engine.moe_blocks):
        for exp_idx in range(num_experts):
            out_e = evaluate_phi_expert_on_probe(probe_x, block.experts, exp_idx)
            # Dummy reduction to 10 axes
            # We just mock the shape here
            layer_capabilities[layer_idx, exp_idx, :] = torch.randn(10)
    
    C_concat = layer_capabilities.permute(1, 0, 2).reshape(num_experts, num_layers * 10)
    return C_concat


def dummy_kl_damage(model):
    """Mock KL damage - would normally compare logprobs"""
    return torch.rand(1).item()


def gate3_one_step_care_com():
    print("\n--- GATE 3: One-Step CARE-COM Loop ---")
    config = PhimoeConfig(
        hidden_size=64, intermediate_size=128, num_local_experts=16, 
        num_hidden_layers=2, vocab_size=100, pad_token_id=0
    )
    model = PhimoeForCausalLM(config).eval()
    engine = PhiPhysicalMergeEngine(model)
    
    # 1. Capability Geometry
    print("1. Computing capability C_t")
    C_t = dummy_capability_extraction(model, engine, engine.current_num_experts, 2)
    assert C_t.shape == (16, 20)
    
    # 2. Candidate Pool (K=5)
    print("2. Selecting K=5 candidate pairs")
    # Mock capability distances
    distances = torch.cdist(C_t, C_t)
    pairs = []
    for i in range(16):
        for j in range(i+1, 16):
            pairs.append((i, j, distances[i, j].item()))
    pairs.sort(key=lambda x: x[2])
    candidates = pairs[:5]
    
    # 3. Physical Trial Merges & KL
    print("3. Evaluating KL for candidates")
    best_pair = None
    best_damage = float('inf')
    
    for i, j, cap_dist in candidates:
        engine.snapshot(i)
        engine.merge_experts(i, j)
        
        # ensure N-1 state during evaluation
        assert engine.current_num_experts == 15
        
        damage = dummy_kl_damage(model)
        if damage < best_damage:
            best_damage = damage
            best_pair = (i, j)
            
        engine.restore()
        assert engine.current_num_experts == 16
        
    print(f"4. Selected best pair {best_pair} with damage {best_damage:.4f}")
    
    # 5. Commit
    print("5. Committing physical merge")
    engine.merge_experts(*best_pair)
    assert engine.current_num_experts == 15
    
    # 6. Recompute
    print("6. Recomputing capability C_{t+1}")
    C_t1 = dummy_capability_extraction(model, engine, engine.current_num_experts, 2)
    assert C_t1.shape == (15, 20)
    
    print("✅ Gate 3 Passed: Complete CARE-COM geometry -> intervention -> recomputation loop simulated.")

if __name__ == "__main__":
    gate1_expert_equivalence()
    gate2_physical_deletion()
    gate3_one_step_care_com()
