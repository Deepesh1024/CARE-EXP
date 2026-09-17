import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from .config import CareComV2Config
from .capability import compute_capability_vectors, pairwise_capability_distances
from .evaluator import collect_baseline_wikitext_logits, evaluate_temporary_pruning

class CapabilityAwareTopKRouter(torch.nn.Module):
    """
    A dedicated capability-aware routing module for the experimental compression model.
    Preserves the original router's linear gate untouched, but performs exact 
    probability mass transfer before top-k selection.
    """
    def __init__(self, original_gate: torch.nn.Module, removed_experts_map: Dict[int, int], top_k: int, capability_redistribution: bool = True):
        super().__init__()
        self.original_gate = original_gate
        self.removed_experts_map = removed_experts_map
        self.top_k = top_k
        self.capability_redistribution = capability_redistribution
        
    def forward(self, hidden_states):
        # 1. Compute logits from untouched original gate
        router_logits = self.original_gate(hidden_states)
        
        # 2. Compute original probability distribution
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 3. Determine ORIGINAL top-k expert indices
        _, original_top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 4. Exact Probability Redistribution (Conditional)
        if self.capability_redistribution:
            for removed_i, dest_j in self.removed_experts_map.items():
                # Identify tokens where removed_i would have been originally selected
                mask_i = (original_top_k_indices == removed_i).any(dim=-1)
                
                # Transfer mass ONLY for those tokens
                routing_weights[mask_i, dest_j] += routing_weights[mask_i, removed_i]
                routing_weights[mask_i, removed_i] = 0.0
                
        # 5. Top-K Selection on the modified probabilities
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 6. Renormalize surviving probabilities (standard OLMoE behavior)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return router_logits, routing_weights, selected_experts

def generate_candidate_pool(distances: Dict[int, Dict[int, float]], usage: Dict[int, float], config: CareComV2Config) -> List[int]:
    redundancy_scores = []
    
    for i, dists in distances.items():
        if not dists:
            continue
        
        nn_dist = min(dists.values())
        u_i = usage.get(i, 0.0)
        
        if config.candidate_scoring == "capability_only":
            score = nn_dist
        elif config.candidate_scoring == "usage_only":
            # Usage only: low usage is highly redundant
            score = u_i
        elif config.candidate_scoring == "capability_plus_usage":
            # Low capability redundancy distance + low usage => higher removal priority (lower score)
            score = nn_dist * (u_i + 1e-6)
        else:
            score = nn_dist
            
        redundancy_scores.append((i, score))
        
    redundancy_scores.sort(key=lambda x: x[1])
    
    pool_size = min(config.candidate_pool_size, len(redundancy_scores))
    return [x[0] for x in redundancy_scores[:pool_size]]

def compress_layer_adaptive(model, moe_blocks, layer_idx: int, df_tokens, eval_chunks, usage_stats: Dict[int, float], config: CareComV2Config):
    print(f"Starting adaptive compression for Layer {layer_idx}")
    
    all_experts = list(range(len(moe_blocks[layer_idx].experts)))
    current_experts = all_experts.copy()
    
    trace_log = []
    
    baseline_logprobs = collect_baseline_wikitext_logits(
        model, eval_chunks, config.eval_batch_size, config.max_eval_batches, config.device
    )
    
    moe_block = moe_blocks[layer_idx]
    
    # Locate original gate and top_k
    original_gate = None
    top_k = None
    for n, m in moe_block.named_modules():
        if m.__class__.__name__ == "OlmoeTopKRouter":
            original_gate = m.weight if hasattr(m, 'weight') else m
            top_k = getattr(m, 'top_k', 8)
            break
            
    if original_gate is None and hasattr(moe_block, 'gate'):
        original_gate = moe_block.gate
        top_k = getattr(moe_block, 'top_k', 8)
        
    # Standard OLMoE uses an internal gate linear layer
    if hasattr(original_gate, 'weight') and not isinstance(original_gate, torch.nn.Linear):
        # The router is often the module containing the weight
        pass
        
    # We create a mapping of permanently removed experts
    removed_map = {}
    
    iteration = 0
    # Initialize capability vectors for all initially retained experts
    C = compute_capability_vectors(model, moe_blocks, df_tokens, layer_idx, current_experts, config.device)
    
    while len(current_experts) > config.target_num_experts:
        start_time = time.time()
        
        distances = pairwise_capability_distances(C, current_experts)
        candidates = generate_candidate_pool(distances, usage_stats, config)
        
        candidate_results = []
        
        for candidate in candidates:
            # Find nearest retained neighbor
            nn_expert = min(distances[candidate].items(), key=lambda x: x[1])[0]
            nn_dist = distances[candidate][nn_expert]
            
            # Temporary mapping
            temp_map = removed_map.copy()
            temp_map[candidate] = nn_expert
            
            # Override routing with exact mass transfer
            temp_router = CapabilityAwareTopKRouter(
                original_gate=original_gate, 
                removed_experts_map=temp_map, 
                top_k=top_k, 
                capability_redistribution=config.capability_redistribution
            )
            
            # Apply temporary router
            if hasattr(moe_block, 'router'):
                original_router_module = moe_block.router
                moe_block.router = temp_router
            else:
                original_router_module = moe_block.gate
                moe_block.gate = temp_router
                
            damage = evaluate_temporary_pruning(
                model, eval_chunks, baseline_logprobs, config.eval_batch_size, config.max_eval_batches, config.device
            )
            
            # Restore original
            if hasattr(moe_block, 'router'):
                moe_block.router = original_router_module
            else:
                moe_block.gate = original_router_module
                
            candidate_results.append({
                "candidate": candidate,
                "capability_distance": float(nn_dist),
                "nearest_neighbor": int(nn_expert),
                "usage": float(usage_stats.get(candidate, 0.0)),
                "damage": float(damage)
            })
            
        # Selection
        best_candidate_info = min(candidate_results, key=lambda x: x["damage"])
        selected_expert = best_candidate_info["candidate"]
        dest_expert = best_candidate_info["nearest_neighbor"]
        
        # Permanently accept operation
        removed_map[selected_expert] = dest_expert
        current_experts.remove(selected_expert)
        
        # Permanently install routing
        perm_router = CapabilityAwareTopKRouter(
            original_gate=original_gate, 
            removed_experts_map=removed_map, 
            top_k=top_k, 
            capability_redistribution=config.capability_redistribution
        )
        if hasattr(moe_block, 'router'):
            moe_block.router = perm_router
        else:
            moe_block.gate = perm_router
            
        # Recompute capabilities for the updated state (if enabled)
        if config.adaptive_recompute:
            C = compute_capability_vectors(model, moe_blocks, df_tokens, layer_idx, current_experts, config.device)
            
        eval_time = time.time() - start_time
        
        iteration_record = {
            "step": iteration,
            "layer": layer_idx,
            "removed_expert": selected_expert,
            "destination_expert": dest_expert,
            "capability_distance": best_candidate_info["capability_distance"],
            "removed_expert_usage": best_candidate_info["usage"],
            "destination_expert_usage": float(usage_stats.get(dest_expert, 0.0)),
            "predicted_redundancy": best_candidate_info["capability_distance"],
            "measured_functional_damage": best_candidate_info["damage"],
            "accepted": True,
            "number_of_remaining_experts": len(current_experts)
        }
        trace_log.append(iteration_record)
        
        if config.save_trace:
            with open(config.trace_path, "w") as f:
                json.dump(trace_log, f, indent=2)
                
        iteration += 1
        
    return model, trace_log
