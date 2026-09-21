import time
import json
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from .capability import compute_global_capability_vectors, pairwise_capability_distances
from .evaluator import collect_current_logits, evaluate_marginal_damage
from .config import CareComV21Config

class PhysicalMergeEngine:
    def __init__(self, model):
        self.model = model
        
        self.moe_blocks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                self.moe_blocks.append(module)
                
        if not self.moe_blocks:
            raise ValueError("No MoE blocks found in the model.")
            
        first_block = self.moe_blocks[0]
        router = getattr(first_block, "gate", getattr(first_block, "router", None))
        self.current_num_experts = router.weight.shape[0]
        
        self._snapshot = None
        
    def snapshot(self):
        """Creates a transactional backup of the exact module lists and weights."""
        self._snapshot = {
            "model_num_experts": getattr(self.model.config, 'num_experts', None),
            "blocks": []
        }
        
        for block in self.moe_blocks:
            router = getattr(block, "gate", getattr(block, "router", None))
            
            # Save original experts list (module list itself)
            original_experts = nn.ModuleList([exp for exp in block.experts])
            
            # Save router weights
            original_router_weight = router.weight.clone().detach()
            
            # Save num_experts attributes across the block recursively
            num_experts_map = []
            for module in block.modules():
                if hasattr(module, 'num_experts'):
                    num_experts_map.append((module, module.num_experts))
                    
            # Deepcopy of the weights that are going to be in-place modified during merge
            # Wait, block.experts[i].gate_proj.weight is copied in-place!
            # We must backup the original parameter data for all experts so we can restore them exactly.
            expert_weights = []
            for exp in block.experts:
                expert_weights.append({
                    "gate": exp.gate_proj.weight.clone().detach(),
                    "up": exp.up_proj.weight.clone().detach(),
                    "down": exp.down_proj.weight.clone().detach()
                })
                
            self._snapshot["blocks"].append({
                "block": block,
                "router": router,
                "experts": original_experts,
                "expert_weights": expert_weights,
                "router_weight": original_router_weight,
                "num_experts_map": num_experts_map
            })
            
    def restore(self):
        """Restores the model precisely to the state in the snapshot."""
        if self._snapshot is None:
            raise RuntimeError("No snapshot to restore from.")
            
        if self._snapshot["model_num_experts"] is not None:
            self.model.config.num_experts = self._snapshot["model_num_experts"]
            
        for b_data in self._snapshot["blocks"]:
            block = b_data["block"]
            router = b_data["router"]
            
            # Restore experts ModuleList
            block.experts = b_data["experts"]
            
            # Restore expert parameter weights that were modified in-place
            for i, exp in enumerate(block.experts):
                w_dict = b_data["expert_weights"][i]
                exp.gate_proj.weight.data.copy_(w_dict["gate"])
                exp.up_proj.weight.data.copy_(w_dict["up"])
                exp.down_proj.weight.data.copy_(w_dict["down"])
            
            # Restore router weight
            router.weight = nn.Parameter(b_data["router_weight"])
            
            # Restore num_experts map
            for module, orig_val in b_data["num_experts_map"]:
                module.num_experts = orig_val
                
        self.current_num_experts = self._snapshot["blocks"][0]["router"].weight.shape[0]
        self._snapshot = None

    @torch.no_grad()
    def merge_experts(self, expert_i: int, expert_j: int):
        """
        Globally merges expert_j into expert_i physically across ALL layers.
        Removes expert_j from the module list.
        """
        if expert_i >= self.current_num_experts or expert_j >= self.current_num_experts:
            raise ValueError(f"Invalid expert indices {expert_i}, {expert_j} for N={self.current_num_experts}")
            
        for block in self.moe_blocks:
            router_module = getattr(block, "gate", getattr(block, "router", None))
            
            # 1. Merge the internal expert MLPs
            gate_new = (block.experts[expert_i].gate_proj.weight.data + block.experts[expert_j].gate_proj.weight.data) / 2.0
            up_new = (block.experts[expert_i].up_proj.weight.data + block.experts[expert_j].up_proj.weight.data) / 2.0
            down_new = (block.experts[expert_i].down_proj.weight.data + block.experts[expert_j].down_proj.weight.data) / 2.0
            
            # 2. Merge router weights
            router_w = router_module.weight.data
            router_new = (router_w[expert_i] + router_w[expert_j]) / 2.0
            
            # Update expert i in-place
            block.experts[expert_i].gate_proj.weight.data.copy_(gate_new)
            block.experts[expert_i].up_proj.weight.data.copy_(up_new)
            block.experts[expert_i].down_proj.weight.data.copy_(down_new)
            
            # 3. Create a new smaller ModuleList omitting expert_j
            keep_indices = [idx for idx in range(len(block.experts)) if idx != expert_j]
            new_experts = nn.ModuleList([block.experts[idx] for idx in keep_indices])
            block.experts = new_experts
            
            new_expert_i = keep_indices.index(expert_i)
            
            # 4. Create smaller router parameter
            new_router_w = router_w[keep_indices].clone()
            new_router_w[new_expert_i] = router_new
            router_module.weight = nn.Parameter(new_router_w)
            
            new_num_experts = len(keep_indices)
            
            # 5. Update recursively the HF `num_experts` properties
            for module in block.modules():
                if hasattr(module, 'num_experts'):
                    module.num_experts = new_num_experts
                    
        # Update config
        if hasattr(self.model.config, 'num_experts'):
            self.model.config.num_experts = new_num_experts
            
        self.current_num_experts = new_num_experts

def generate_candidate_pool(distances: Dict[int, Dict[int, float]], pool_size: int) -> List[Tuple[int, int]]:
    """
    Generates unique (i, j) candidate pairs based on closest capability distance.
    """
    pairs = []
    seen = set()
    
    for i in distances.keys():
        for j in distances[i].keys():
            if i != j:
                pair = tuple(sorted((i, j)))
                if pair not in seen:
                    pairs.append((*pair, distances[i][j]))
                    seen.add(pair)
                    
    pairs.sort(key=lambda x: x[2])
    
    selected_pairs = [(x[0], x[1]) for x in pairs[:pool_size]]
    return selected_pairs

def compress_adaptive(model, engine: PhysicalMergeEngine, df_tokens, eval_chunks, config: CareComV21Config):
    trace_log = []
    
    while engine.current_num_experts > config.trajectory[-1]:
        start_time = time.time()
        
        target_experts = engine.current_num_experts - 1
        print(f"\n--- Compression Step: {engine.current_num_experts} -> {target_experts} ---")
        
        # 1. Measure Current Capability State C_t
        t0 = time.time()
        C = compute_global_capability_vectors(model, engine.moe_blocks, df_tokens, device=config.device)
        capability_time = time.time() - t0
        
        distances = pairwise_capability_distances(C)
        
        # 2. Generate Candidate Pairs
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
            engine.snapshot()
            engine.merge_experts(i, j)
            
            damage = evaluate_marginal_damage(
                model, eval_chunks, current_logprobs, config.eval_batch_size, config.max_eval_batches, config.device
            )
            
            engine.restore()
            
            candidate_results.append({
                "pair": (i, j),
                "capability_distance": float(distances[i][j]),
                "marginal_damage": float(damage)
            })
            
        eval_time = time.time() - t1
            
        # 5. Select ONE merge (argmin D)
        t2 = time.time()
        best_candidate = min(candidate_results, key=lambda x: x["marginal_damage"])
        selected_i, selected_j = best_candidate["pair"]
        
        # 6. Perform Permanent Merge
        engine.merge_experts(selected_i, selected_j)
        merge_time = time.time() - t2
        
        total_time = time.time() - start_time
        
        record = {
            "step": 64 - target_experts,
            "experts_before": target_experts + 1,
            "experts_after": target_experts,
            "selected_pair": [selected_i, selected_j],
            "capability_distance": best_candidate["capability_distance"],
            "functional_damage": best_candidate["marginal_damage"],
            "candidate_pool_size": config.candidate_pool_size,
            "capability_recompute_time": capability_time,
            "evaluation_time": eval_time,
            "merge_time": merge_time,
            "total_step_time": total_time,
            "all_candidates": candidate_results
        }
        
        print(f"Merged ({selected_i}, {selected_j}) | Damage: {best_candidate['marginal_damage']:.4f} | Time: {total_time:.2f}s")
        
        trace_log.append(record)
        
        # Save trace
        with open(config.trace_path, "w") as f:
            json.dump(trace_log, f, indent=2)
            
        if engine.current_num_experts in config.checkpoints:
            print(f"Checkpoint {engine.current_num_experts} reached.")
            
    return model, trace_log
