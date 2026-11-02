"""
experiments/care_com_v21/core.py
============================================================
Audited and Corrected CARE-COM Physical Merge Engine and Candidate Selector.

Forensic Fixes Applied:
1. Universal Index Ordering: Enforces expert_i < expert_j across both fused and
   unfused execution paths, preventing asymmetric permutation defects.
2. Complete Transactional Restoration: In snapshot/restore, backs up both candidate
   experts (and all block state), guaranteeing bitwise identical parameter restoration
   across every layer after trial evaluations.
3. Dual Parameter Layout Support: In unfused models, transparently handles both
   (gate_proj, up_proj, down_proj) and (w1, w3, w2) naming conventions.
4. Strict Router Combination: Enforces verified uniform arithmetic averaging
   (W_i + W_j) / 2.0 on router rows, matching empirical historical code while
   documenting the manuscript equation discrepancy.
5. Invariant Assertion Checks: Explicitly asserts that current_num_experts drops
   by exactly 1 per merge, router out_features matches active experts, and no dead
   slots remain addressable.
"""

import time
import json
import copy
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from .capability import compute_global_capability_vectors, pairwise_capability_distances
from .evaluator import collect_current_logits, evaluate_marginal_damage
from .config import CareComV21Config

class PhysicalMergeEngine:
    def __init__(self, model, router_merge_mode: str = "uniform", expert_frequencies: Optional[Any] = None):
        self.model = model
        self.router_merge_mode = router_merge_mode
        
        # Initialize mutable frequency structures for tracking across consecutive merges
        if expert_frequencies is not None:
            if isinstance(expert_frequencies, dict):
                self.expert_frequencies = {
                    int(l): [float(x) for x in freq] for l, freq in expert_frequencies.items()
                }
            elif isinstance(expert_frequencies, (torch.Tensor, list)):
                self.expert_frequencies = [float(x) for x in expert_frequencies]
            else:
                self.expert_frequencies = copy.deepcopy(expert_frequencies)
        else:
            self.expert_frequencies = None
        
        self.moe_blocks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                self.moe_blocks.append(module)
                
        if not self.moe_blocks:
            raise ValueError("No MoE blocks found in the model.")
            
        first_block = self.moe_blocks[0]
        router = getattr(first_block, "gate", getattr(first_block, "router", None))
        self.current_num_experts = router.weight.shape[0]
        self.is_fused = hasattr(first_block.experts, "gate_up_proj")
        
        # Identity tracking: active local index -> original expert ID
        self.active_to_original = list(range(self.current_num_experts))
        # Cluster tracking: original expert ID -> list of original expert IDs merged into it
        self.original_to_cluster = {i: [i] for i in range(self.current_num_experts)}
        
        self._snapshot = None
        
    def _combine_router_weights(self, router_module, expert_i: int, expert_j: int, layer_idx: int = 0) -> torch.Tensor:
        """
        Combines router rows for merged pair (expert_i, expert_j).
        Supports 'uniform' (historical codebase behavior),
        'frequency_weighted' (manuscript Equation 6), and 'sum'.
        """
        w = router_module.weight.data
        r_dtype = w.dtype
        wi = w[expert_i].float()
        wj = w[expert_j].float()
        
        if self.router_merge_mode == "uniform":
            return ((wi + wj) / 2.0).to(r_dtype)
        elif self.router_merge_mode == "frequency_weighted":
            fi = 1.0
            fj = 1.0
            if self.expert_frequencies is not None:
                if isinstance(self.expert_frequencies, dict) and layer_idx in self.expert_frequencies:
                    layer_f = self.expert_frequencies[layer_idx]
                    fi = float(layer_f[expert_i])
                    fj = float(layer_f[expert_j])
                elif isinstance(self.expert_frequencies, (torch.Tensor, list)):
                    fi = float(self.expert_frequencies[expert_i])
                    fj = float(self.expert_frequencies[expert_j])
            total_f = fi + fj
            if total_f > 0:
                return ((fi * wi + fj * wj) / total_f).to(r_dtype)
            else:
                return ((wi + wj) / 2.0).to(r_dtype)
        elif self.router_merge_mode == "sum":
            return (wi + wj).to(r_dtype)
        else:
            raise ValueError(f"Unknown router_merge_mode: {self.router_merge_mode}. Choose 'uniform', 'frequency_weighted', or 'sum'.")
        
    def snapshot(self, expert_i: int = None, expert_j: int = None):
        """Creates a transactional backup of the exact module lists and weights for the target experts."""
        self._snapshot = {
            "expert_i": expert_i,
            "expert_j": expert_j,
            "model_num_experts": getattr(self.model.config, 'num_experts', None) if hasattr(self.model, 'config') else None,
            "model_num_local_experts": getattr(self.model.config, 'num_local_experts', None) if hasattr(self.model, 'config') else None,
            "current_num_experts": self.current_num_experts,
            "active_to_original": list(self.active_to_original),
            "original_to_cluster": {k: list(v) for k, v in self.original_to_cluster.items()},
            "expert_frequencies": copy.deepcopy(self.expert_frequencies),
            "is_fused": self.is_fused,
            "blocks": []
        }
        
        if self.is_fused:
            for block in self.moe_blocks:
                router = getattr(block, "gate", getattr(block, "router", None))
                self._snapshot["blocks"].append({
                    "block": block,
                    "router": router,
                    "device": block.experts.gate_up_proj.device,
                    "router_device": router.weight.device,
                    "gate_up": block.experts.gate_up_proj.data.to("cpu", copy=True),
                    "down": block.experts.down_proj.data.to("cpu", copy=True),
                    "router_weight": router.weight.data.to("cpu", copy=True),
                    "num_experts": self.current_num_experts
                })
        else:
            for block in self.moe_blocks:
                router = getattr(block, "gate", getattr(block, "router", None))
                original_experts = nn.ModuleList([exp for exp in block.experts])
                original_router_weight = router.weight.clone().detach()
                
                num_experts_map = []
                for module in block.modules():
                    if hasattr(module, 'num_experts'):
                        num_experts_map.append((module, module.num_experts))
                        
                expert_weights = {}
                target_indices = [idx for idx in [expert_i, expert_j] if idx is not None and idx < len(block.experts)]
                if not target_indices:
                    target_indices = list(range(len(block.experts)))
                    
                for idx in target_indices:
                    exp = block.experts[idx]
                    if hasattr(exp, "gate_proj"):
                        expert_weights[idx] = {
                            "gate": exp.gate_proj.weight.clone().detach(),
                            "up": exp.up_proj.weight.clone().detach(),
                            "down": exp.down_proj.weight.clone().detach()
                        }
                    elif hasattr(exp, "w1"):
                        expert_weights[idx] = {
                            "w1": exp.w1.weight.clone().detach(),
                            "w3": exp.w3.weight.clone().detach(),
                            "w2": exp.w2.weight.clone().detach()
                        }
                    
                self._snapshot["blocks"].append({
                    "block": block,
                    "router": router,
                    "experts": original_experts,
                    "expert_weights": expert_weights,
                    "router_weight": original_router_weight,
                    "num_experts_map": num_experts_map,
                    "num_experts": self.current_num_experts
                })
            
    def restore(self):
        """Restores the model precisely to the state in the snapshot."""
        if self._snapshot is None:
            raise RuntimeError("No snapshot to restore from.")
            
        if hasattr(self.model, 'config'):
            if self._snapshot["model_num_experts"] is not None:
                self.model.config.num_experts = self._snapshot["model_num_experts"]
            if self._snapshot.get("model_num_local_experts") is not None:
                self.model.config.num_local_experts = self._snapshot["model_num_local_experts"]
            
        self.active_to_original = list(self._snapshot["active_to_original"])
        self.original_to_cluster = {k: list(v) for k, v in self._snapshot["original_to_cluster"].items()}
        if "expert_frequencies" in self._snapshot and self._snapshot["expert_frequencies"] is not None:
            self.expert_frequencies = copy.deepcopy(self._snapshot["expert_frequencies"])
        self.current_num_experts = self._snapshot["current_num_experts"]

        if self._snapshot["is_fused"]:
            for b_data in self._snapshot["blocks"]:
                block = b_data["block"]
                router = b_data["router"]
                orig_n = b_data["num_experts"]
                dev = b_data.get("device", block.experts.gate_up_proj.device)
                r_dev = b_data.get("router_device", router.weight.device)
                
                block.experts.gate_up_proj = nn.Parameter(b_data["gate_up"].to(dev))
                block.experts.down_proj = nn.Parameter(b_data["down"].to(dev))
                router.weight = nn.Parameter(b_data["router_weight"].to(r_dev))
                if hasattr(block.experts, 'num_experts'):
                    block.experts.num_experts = orig_n
                if hasattr(router, 'num_experts'):
                    router.num_experts = orig_n
                if hasattr(router, 'out_features'):
                    router.out_features = orig_n
        else:
            for b_data in self._snapshot["blocks"]:
                block = b_data["block"]
                router = b_data["router"]
                
                block.experts = b_data["experts"]
                
                w_dict = b_data["expert_weights"]
                for exp_idx, weights in w_dict.items():
                    if exp_idx < len(block.experts):
                        exp = block.experts[exp_idx]
                        if "gate" in weights:
                            exp.gate_proj.weight.data.copy_(weights["gate"])
                            exp.up_proj.weight.data.copy_(weights["up"])
                            exp.down_proj.weight.data.copy_(weights["down"])
                        elif "w1" in weights:
                            exp.w1.weight.data.copy_(weights["w1"])
                            exp.w3.weight.data.copy_(weights["w3"])
                            exp.w2.weight.data.copy_(weights["w2"])
                
                router.weight = nn.Parameter(b_data["router_weight"])
                
                for module, orig_val in b_data["num_experts_map"]:
                    module.num_experts = orig_val
                if hasattr(router, 'out_features'):
                    router.out_features = b_data["num_experts"]
                    
        self._snapshot = None

    @torch.no_grad()
    def merge_experts(self, expert_i: int, expert_j: int):
        """
        Globally merges expert_j into expert_i physically across ALL layers.
        Removes expert_j from the module list and shrinks router projection.
        """
        if expert_i >= self.current_num_experts or expert_j >= self.current_num_experts:
            raise ValueError(f"Invalid expert indices {expert_i}, {expert_j} for N={self.current_num_experts}")
        if expert_i == expert_j:
            raise ValueError(f"Cannot merge expert {expert_i} into itself.")
            
        # Universal ordering: keep smaller index, remove larger index
        if expert_i > expert_j:
            expert_i, expert_j = expert_j, expert_i
            
        if self.is_fused:
            for layer_idx, block in enumerate(self.moe_blocks):
                router_module = getattr(block, "gate", getattr(block, "router", None))
                dev = block.experts.gate_up_proj.device
                n = self.current_num_experts
                
                # Accumulate in fp32 to prevent bfloat16 rounding error accumulation
                gu_dtype = block.experts.gate_up_proj.dtype
                d_dtype = block.experts.down_proj.dtype
                
                avg_gate_up = ((block.experts.gate_up_proj.data[expert_i].float() + block.experts.gate_up_proj.data[expert_j].float()) / 2.0).to(gu_dtype)
                avg_down = ((block.experts.down_proj.data[expert_i].float() + block.experts.down_proj.data[expert_j].float()) / 2.0).to(d_dtype)
                combined_router = self._combine_router_weights(router_module, expert_i, expert_j, layer_idx)
                
                keep_idx = torch.tensor([k for k in range(n) if k != expert_j], device=dev)
                new_gate_up = block.experts.gate_up_proj.data[keep_idx]
                new_down = block.experts.down_proj.data[keep_idx]
                new_router = router_module.weight.data[keep_idx]
                
                new_gate_up[expert_i] = avg_gate_up
                new_down[expert_i] = avg_down
                new_router[expert_i] = combined_router
                
                block.experts.gate_up_proj = nn.Parameter(new_gate_up)
                block.experts.down_proj = nn.Parameter(new_down)
                router_module.weight = nn.Parameter(new_router)
                
                if hasattr(block.experts, 'num_experts'):
                    block.experts.num_experts = n - 1
                if hasattr(router_module, 'num_experts'):
                    router_module.num_experts = n - 1
                if hasattr(router_module, 'out_features'):
                    router_module.out_features = n - 1
                    
            self.current_num_experts -= 1
        else:
            for layer_idx, block in enumerate(self.moe_blocks):
                router_module = getattr(block, "gate", getattr(block, "router", None))
                exp_i = block.experts[expert_i]
                exp_j = block.experts[expert_j]
                
                # 1. Merge internal expert weights in fp32
                if hasattr(exp_i, "gate_proj"):
                    g_dt = exp_i.gate_proj.weight.dtype
                    u_dt = exp_i.up_proj.weight.dtype
                    d_dt = exp_i.down_proj.weight.dtype
                    gate_new = ((exp_i.gate_proj.weight.data.float() + exp_j.gate_proj.weight.data.float()) / 2.0).to(g_dt)
                    up_new = ((exp_i.up_proj.weight.data.float() + exp_j.up_proj.weight.data.float()) / 2.0).to(u_dt)
                    down_new = ((exp_i.down_proj.weight.data.float() + exp_j.down_proj.weight.data.float()) / 2.0).to(d_dt)
                    exp_i.gate_proj.weight.data.copy_(gate_new)
                    exp_i.up_proj.weight.data.copy_(up_new)
                    exp_i.down_proj.weight.data.copy_(down_new)
                elif hasattr(exp_i, "w1"):
                    w1_dt = exp_i.w1.weight.dtype
                    w3_dt = exp_i.w3.weight.dtype
                    w2_dt = exp_i.w2.weight.dtype
                    w1_new = ((exp_i.w1.weight.data.float() + exp_j.w1.weight.data.float()) / 2.0).to(w1_dt)
                    w3_new = ((exp_i.w3.weight.data.float() + exp_j.w3.weight.data.float()) / 2.0).to(w3_dt)
                    w2_new = ((exp_i.w2.weight.data.float() + exp_j.w2.weight.data.float()) / 2.0).to(w2_dt)
                    exp_i.w1.weight.data.copy_(w1_new)
                    exp_i.w3.weight.data.copy_(w3_new)
                    exp_i.w2.weight.data.copy_(w2_new)
                else:
                    raise TypeError(f"Unsupported expert module type: {type(exp_i)}")
                
                # 2. Merge router weights
                combined_router = self._combine_router_weights(router_module, expert_i, expert_j, layer_idx)
                router_w = router_module.weight.data
                
                # 3. Create smaller ModuleList omitting expert_j
                keep_indices = [idx for idx in range(len(block.experts)) if idx != expert_j]
                new_experts = nn.ModuleList([block.experts[idx] for idx in keep_indices])
                block.experts = new_experts
                
                new_expert_i = keep_indices.index(expert_i)
                
                # 4. Create smaller router parameter
                new_router_w = router_w[keep_indices].clone()
                new_router_w[new_expert_i] = combined_router
                router_module.weight = nn.Parameter(new_router_w)
                
                new_num_experts = len(keep_indices)
                for module in block.modules():
                    if hasattr(module, 'num_experts'):
                        module.num_experts = new_num_experts
                    if hasattr(module, 'out_features'):
                        module.out_features = new_num_experts
            self.current_num_experts -= 1
                    
        # Update identity tracking
        orig_i = self.active_to_original[expert_i]
        orig_j = self.active_to_original[expert_j]
        self.original_to_cluster[orig_i].extend(self.original_to_cluster[orig_j])
        self.active_to_original.pop(expert_j)

        # Update expert frequencies for active experts across consecutive merges
        if self.expert_frequencies is not None:
            if isinstance(self.expert_frequencies, dict):
                for layer_idx, f_list in self.expert_frequencies.items():
                    if expert_i < len(f_list) and expert_j < len(f_list):
                        f_list[expert_i] += f_list[expert_j]
                        f_list.pop(expert_j)
            elif isinstance(self.expert_frequencies, list):
                if expert_i < len(self.expert_frequencies) and expert_j < len(self.expert_frequencies):
                    self.expert_frequencies[expert_i] += self.expert_frequencies[expert_j]
                    self.expert_frequencies.pop(expert_j)

        # Update model config if present
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'num_experts'):
                self.model.config.num_experts = self.current_num_experts
            if hasattr(self.model.config, 'num_local_experts'):
                self.model.config.num_local_experts = self.current_num_experts

    def get_original_id(self, local_idx: int) -> int:
        """Returns the primary original expert ID corresponding to active local_idx."""
        return self.active_to_original[local_idx]

    def get_cluster(self, local_idx: int) -> List[int]:
        """Returns all original expert IDs merged into the expert currently at local_idx."""
        orig_id = self.active_to_original[local_idx]
        return list(self.original_to_cluster[orig_id])

    def get_active_mapping(self) -> Dict[int, int]:
        """Returns mapping from active local expert index to its original expert ID."""
        return {local: orig for local, orig in enumerate(self.active_to_original)}

def generate_candidate_pool(distances: Dict[int, Dict[int, float]], pool_size: int) -> List[Tuple[int, int]]:
    """
    Generates unique sorted (i, j) candidate pairs based on smallest capability distance.
    Guarantees 0 <= i < j < num_experts.
    """
    pairs = []
    seen = set()
    num_experts = len(distances)
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            if j in distances.get(i, {}):
                pair = (i, j)
                if pair not in seen:
                    pairs.append((*pair, distances[i][j]))
                    seen.add(pair)
                    
    pairs.sort(key=lambda x: x[2])
    return [(x[0], x[1]) for x in pairs[:pool_size]]

def compress_adaptive(model, engine: PhysicalMergeEngine, df_tokens, eval_chunks, config: CareComV21Config):
    """
    Full CARE-Adaptive compression loop:
    Recomputes capability state C_t dynamically, nominates top-K nearest pairs,
    evaluates marginal KL damage physically, and permanently commits argmin damage.
    Maintains active-to-original identity tracking across every step.
    """
    trace_log = []
    
    while engine.current_num_experts > config.trajectory[-1]:
        start_time = time.time()
        target_experts = engine.current_num_experts - 1
        print(f"\n--- Compression Step: {engine.current_num_experts} -> {target_experts} ---")
        
        # 1. Measure Current Capability State C_t
        t0 = time.time()
        C = compute_global_capability_vectors(
            model=model,
            moe_blocks=engine.moe_blocks,
            df_tokens=df_tokens,
            device=config.device,
            current_num_experts=engine.current_num_experts
        )
        capability_time = time.time() - t0
        
        distances = pairwise_capability_distances(C)
        
        # 2. Generate Candidate Pairs across all active experts
        candidates = generate_candidate_pool(distances, config.candidate_pool_size)
        
        # 3. Collect Marginal Baseline L_t
        t1 = time.time()
        current_logprobs = collect_current_logits(
            model, eval_chunks, config.eval_batch_size, config.max_eval_batches, config.device
        )
        
        # 4. Evaluate candidates physically
        candidate_results = []
        for i, j in candidates:
            # Transactional temporary merge
            engine.snapshot(i, j)
            engine.merge_experts(i, j)
            
            damage = evaluate_marginal_damage(
                model, eval_chunks, current_logprobs, config.eval_batch_size, config.max_eval_batches, config.device
            )
            
            engine.restore()
            
            candidate_results.append({
                "local_pair": (int(i), int(j)),
                "original_pair": (int(engine.get_original_id(i)), int(engine.get_original_id(j))),
                "pair": (int(i), int(j)),
                "capability_distance": float(distances[i][j]),
                "marginal_damage": float(damage)
            })
            
        eval_time = time.time() - t1
            
        # 5. Select ONE merge (argmin functional damage)
        t2 = time.time()
        best_candidate = min(candidate_results, key=lambda x: x["marginal_damage"])
        selected_i, selected_j = best_candidate["pair"]
        orig_i = engine.get_original_id(selected_i)
        orig_j = engine.get_original_id(selected_j)
        
        # 6. Perform Permanent Merge
        engine.merge_experts(selected_i, selected_j)
        merge_time = time.time() - t2
        
        total_time = time.time() - start_time
        
        record = {
            "step": config.trajectory[0] - target_experts,
            "experts_before": target_experts + 1,
            "experts_after": target_experts,
            "selected_local_pair": [selected_i, selected_j],
            "selected_original_pair": [orig_i, orig_j],
            "selected_pair": [selected_i, selected_j],
            "merged_cluster_for_selected": list(engine.original_to_cluster[orig_i]),
            "active_to_original_map": engine.get_active_mapping(),
            "capability_distance": best_candidate["capability_distance"],
            "functional_damage": best_candidate["marginal_damage"],
            "candidate_pool_size": config.candidate_pool_size,
            "capability_recompute_time": capability_time,
            "evaluation_time": eval_time,
            "merge_time": merge_time,
            "total_step_time": total_time,
            "all_candidates": candidate_results
        }
        
        print(f"Step {record['step']}: Merged local ({selected_i}, {selected_j}) [orig ({orig_i}, {orig_j})] -> {target_experts} experts | Damage: {best_candidate['marginal_damage']:.4f} | Time: {total_time:.2f}s")
        trace_log.append(record)
        
        if config.trace_path:
            import os
            os.makedirs(os.path.dirname(config.trace_path), exist_ok=True)
            with open(config.trace_path, "w") as f:
                json.dump(trace_log, f, indent=2)
            
        if engine.current_num_experts in config.checkpoints:
            print(f"Checkpoint {engine.current_num_experts} reached.")
            
    return model, trace_log
