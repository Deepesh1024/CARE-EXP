import time
import json
import random
import torch
from typing import List, Dict, Any, Tuple

from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage
from experiments.care_com_v21.capability import compute_global_capability_vectors, pairwise_capability_distances
from experiments.care_com_v22.config import CareComV22Config

def compute_ppl(model, eval_chunks, config: CareComV22Config):
    """Computes perplexity on the evaluation subset."""
    model.eval()
    nlls = []
    batches = min(len(eval_chunks) // config.ppl_batch_size, config.max_ppl_batches)
    
    with torch.no_grad():
        for i in range(batches):
            batch = eval_chunks[i * config.ppl_batch_size : (i + 1) * config.ppl_batch_size]
            input_ids = torch.stack([x["input_ids"] for x in batch]).to(config.device)
            attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(config.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # HF loss is mean NLL over all non-ignored tokens in the batch
            nlls.append(outputs.loss.item())
            
    avg_nll = sum(nlls) / len(nlls) if nlls else 0
    return torch.exp(torch.tensor(avg_nll)).item()

def run_random_baseline(model, engine: PhysicalMergeEngine, eval_chunks, config: CareComV22Config, seed: int):
    """
    Runs the random physical merging baseline.
    Does NOT use capability distance. Evaluates marginal KL for logging only.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    trace_log = []
    ppl_log = {}
    
    # Initial PPL at 64
    if engine.current_num_experts in config.checkpoints:
        ppl_log[engine.current_num_experts] = compute_ppl(model, eval_chunks, config)
        print(f"PPL @ {engine.current_num_experts}: {ppl_log[engine.current_num_experts]:.4f}")
    
    for target_experts in config.trajectory[1:]: # 63, 62, ...
        start_time = time.time()
        print(f"\n[Random Seed {seed}] Compression Step: {engine.current_num_experts} -> {target_experts}")
        
        # 1. Randomly select valid active pair
        num_active = engine.current_num_experts
        i = random.randint(0, num_active - 1)
        j = random.randint(0, num_active - 1)
        while i == j:
            j = random.randint(0, num_active - 1)
        
        selected_i, selected_j = min(i, j), max(i, j)
        
        # 2. Collect current logits for logging baseline damage
        t1 = time.time()
        current_logprobs = collect_current_logits(
            model, eval_chunks, config.eval_batch_size, config.max_eval_batches, config.device
        )
        
        # 3. Transactionally evaluate damage of the random choice (strictly for logging)
        engine.snapshot(selected_i)
        engine.merge_experts(selected_i, selected_j)
        damage = evaluate_marginal_damage(
            model, eval_chunks, current_logprobs, config.eval_batch_size, config.max_eval_batches, config.device
        )
        engine.restore()
        eval_time = time.time() - t1
        
        # 4. Permanently merge
        t2 = time.time()
        engine.merge_experts(selected_i, selected_j)
        merge_time = time.time() - t2
        
        total_time = time.time() - start_time
        
        record = {
            "step": 64 - target_experts,
            "experts_before": target_experts + 1,
            "experts_after": target_experts,
            "selected_pair": [selected_i, selected_j],
            "functional_damage": float(damage),
            "evaluation_time": eval_time,
            "merge_time": merge_time,
            "total_step_time": total_time,
            "seed": seed
        }
        
        print(f"Randomly Merged ({selected_i}, {selected_j}) | Damage: {damage:.4f} | Time: {total_time:.2f}s")
        trace_log.append(record)
        
        if target_experts in config.checkpoints:
            ppl_log[target_experts] = compute_ppl(model, eval_chunks, config)
            print(f"PPL @ {target_experts}: {ppl_log[target_experts]:.4f}")
        
    return trace_log, ppl_log

def get_frozen_ranking(C):
    """Sorts all pairs based on capability distance from the initial C_64 state."""
    distances = pairwise_capability_distances(C)
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
    return pairs

def map_original_indices_to_current(original_i, original_j, removed_experts):
    """
    Given an original pair from N=64, map it to the current re-indexed module list.
    If either expert was removed, return None.
    """
    if original_i in removed_experts or original_j in removed_experts:
        return None
        
    current_i = original_i - sum(1 for r in removed_experts if r < original_i)
    current_j = original_j - sum(1 for r in removed_experts if r < original_j)
    
    return min(current_i, current_j), max(current_i, current_j)

def run_static_baseline(model, engine: PhysicalMergeEngine, df_tokens, eval_chunks, config: CareComV22Config):
    """
    Runs the static v1 CARE-COM baseline.
    Computes capability EXACTLY ONCE at 64 experts. 
    Iterates through the frozen ranking, skipping invalid pairs.
    """
    trace_log = []
    ppl_log = {}
    
    # Initial PPL at 64
    if engine.current_num_experts in config.checkpoints:
        ppl_log[engine.current_num_experts] = compute_ppl(model, eval_chunks, config)
        print(f"PPL @ {engine.current_num_experts}: {ppl_log[engine.current_num_experts]:.4f}")
    
    print("\n[Static v1] Computing capability geometry ONCE at N=64...")
    t0 = time.time()
    C_64 = compute_global_capability_vectors(model, engine.moe_blocks, df_tokens, device=config.device)
    capability_time = time.time() - t0
    
    frozen_ranking = get_frozen_ranking(C_64)
    removed_experts = set()
    ranking_idx = 0
    
    for target_experts in config.trajectory[1:]:
        start_time = time.time()
        print(f"\n[Static v1] Compression Step: {engine.current_num_experts} -> {target_experts}")
        
        # 1. Find the highest valid pair in the frozen ranking
        valid_pair = None
        while ranking_idx < len(frozen_ranking):
            orig_i, orig_j, cap_dist = frozen_ranking[ranking_idx]
            mapped_pair = map_original_indices_to_current(orig_i, orig_j, removed_experts)
            
            if mapped_pair is not None:
                valid_pair = {
                    "original": (orig_i, orig_j),
                    "current": mapped_pair,
                    "capability_distance": cap_dist,
                    "rank": ranking_idx
                }
                # Once we select it, it will be merged, so we advance the index for next time
                ranking_idx += 1
                break
                
            ranking_idx += 1
            
        if valid_pair is None:
            raise RuntimeError("Exhausted all pairs in frozen ranking without reaching target experts!")
            
        selected_i, selected_j = valid_pair["current"]
        orig_i, orig_j = valid_pair["original"]
        
        # 2. Collect current logits for logging baseline damage against M_t
        t1 = time.time()
        current_logprobs = collect_current_logits(
            model, eval_chunks, config.eval_batch_size, config.max_eval_batches, config.device
        )
        
        # 3. Transactionally evaluate damage of the static choice (strictly for logging comparison)
        engine.snapshot(selected_i)
        engine.merge_experts(selected_i, selected_j)
        damage = evaluate_marginal_damage(
            model, eval_chunks, current_logprobs, config.eval_batch_size, config.max_eval_batches, config.device
        )
        engine.restore()
        eval_time = time.time() - t1
        
        # 4. Permanently merge
        t2 = time.time()
        engine.merge_experts(selected_i, selected_j)
        merge_time = time.time() - t2
        
        # Record the removal so future original indices map correctly
        removed_experts.add(orig_j)
        
        total_time = time.time() - start_time
        
        record = {
            "step": 64 - target_experts,
            "experts_before": target_experts + 1,
            "experts_after": target_experts,
            "selected_pair": [selected_i, selected_j],
            "original_pair": [orig_i, orig_j],
            "frozen_rank": valid_pair["rank"],
            "capability_distance": float(valid_pair["capability_distance"]),
            "functional_damage": float(damage),
            "capability_recompute_time": capability_time if target_experts == 63 else 0.0,
            "evaluation_time": eval_time,
            "merge_time": merge_time,
            "total_step_time": total_time
        }
        
        print(f"Static Merged original ({orig_i}, {orig_j}) -> current ({selected_i}, {selected_j}) | Damage: {damage:.4f} | Time: {total_time:.2f}s")
        trace_log.append(record)
        
        if target_experts in config.checkpoints:
            ppl_log[target_experts] = compute_ppl(model, eval_chunks, config)
            print(f"PPL @ {target_experts}: {ppl_log[target_experts]:.4f}")
        
    return trace_log, ppl_log
