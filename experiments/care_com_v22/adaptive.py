import time
import torch
from typing import List, Dict, Any

from experiments.care_com_v21.core import PhysicalMergeEngine, generate_candidate_pool
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage
from experiments.care_com_v21.capability import compute_global_capability_vectors, pairwise_capability_distances
from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.baselines import compute_ppl

def run_adaptive_baseline(model, engine: PhysicalMergeEngine, df_tokens, eval_chunks, config: CareComV22Config):
    """
    Runs the adaptive v2.1 CARE-COM baseline.
    Recomputes capability at every step and dynamically selects candidates based on marginal KL.
    """
    trace_log = []
    ppl_log = {}
    
    # Initial PPL at 64
    if engine.current_num_experts in config.checkpoints:
        ppl_log[engine.current_num_experts] = compute_ppl(model, eval_chunks, config)
        print(f"PPL @ {engine.current_num_experts}: {ppl_log[engine.current_num_experts]:.4f}")
    
    for target_experts in config.trajectory[1:]:
        start_time = time.time()
        print(f"\n[Adaptive v2.1] Compression Step: {engine.current_num_experts} -> {target_experts}")
        
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
            engine.snapshot(i)
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
        
        print(f"Adaptive Merged ({selected_i}, {selected_j}) | Damage: {best_candidate['marginal_damage']:.4f} | Time: {total_time:.2f}s")
        trace_log.append(record)
        
        if target_experts in config.checkpoints:
            ppl_log[target_experts] = compute_ppl(model, eval_chunks, config)
            print(f"PPL @ {target_experts}: {ppl_log[target_experts]:.4f}")
        
    return trace_log, ppl_log
