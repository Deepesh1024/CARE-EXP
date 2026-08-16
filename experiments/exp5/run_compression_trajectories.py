
import os
import json
import time
import copy
import argparse
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from xgboost import XGBRegressor
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lm_eval
import lm_eval.tasks
from lm_eval.models.huggingface import HFLM

from compression_core import MoECompressionEngine, greedy_conflict_resolution

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    DEVICE = "mps"

TIER_1_LEVELS = [64, 56, 48, 40, 32, 24, 16]
TIER_2_LEVELS = [64, 48, 32, 16]

def load_base_model():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32, 
        device_map=DEVICE if DEVICE == "cuda:0" else None
    )
    if DEVICE == "mps":
        model = model.to(DEVICE)
    return model, tokenizer

def evaluate_tier1(model, tokenizer):
    print("  -> Evaluating Tier 1 (PPL, Efficiency)")
    # 1. PPL
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    # 2. Efficiency
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 512)).to(DEVICE)
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
        
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start_t = time.perf_counter()
            _ = model(dummy_input)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elif torch.backends.mps.is_available(): torch.mps.synchronize()
            end_t = time.perf_counter()
            latencies.append((end_t - start_t) * 1000)
            
    latencies = np.array(latencies)
    throughput = 512 / (latencies / 1000)
    
    # 3. Parameter count (approximate based on active experts)
    engine = MoECompressionEngine(model)
    active = len(engine.active_experts)
    # Each expert has 2048*2048 + 2048*1024 params approx
    expert_params = (2048*2048 + 2048*1024) * active
    
    return {
        "wikitext_2_ppl": ppl,
        "latency_mean_ms": float(np.mean(latencies)),
        "throughput_mean_tps": float(np.mean(throughput)),
        "active_experts": active,
        "expert_params": expert_params
    }

def evaluate_tier2(model, tokenizer, seed=42):
    print("  -> Evaluating Tier 2 (ARC, MMLU)")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, device=DEVICE)
    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["arc_challenge", "mmlu"],
        num_fewshot=0,
        batch_size="auto",
        task_manager=task_manager,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    return {
        "arc_challenge_acc": results['results']['arc_challenge']['acc,none'],
        "mmlu_acc": results['results']['mmlu']['acc,none']
    }

def get_parameter_similarity_ranking(engine):
    active = engine.active_experts
    pairs = []
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            ei = active[i]
            ej = active[j]
            pi = engine.get_expert_params(ei)
            pj = engine.get_expert_params(ej)
            # L2 distance
            dist = torch.norm(pi - pj, p=2).item()
            pairs.append((ei, ej, dist))
    # Sort by smallest distance first
    pairs.sort(key=lambda x: x[2])
    return pairs

def get_care_com_ranking(engine, active_experts):
    # This requires extracting the 11 local features for the CURRENT state
    # This is highly complex for iterative mode because it requires running calibration data again.
    # For one-shot, we can just use the pre-computed predictions.
    
    # To keep this script self-contained and execute what we can, we load the offline trained model.
    # For one-shot, we evaluate it once.
    # In a real environment, iterative CARE requires a full feature extraction pipeline call here.
    
    model_path = "results/exp5/care_com_model.json"
    scaler_path = "results/exp5/care_com_scaler.joblib"
    exp1_path = "results/exp1/output.json"
    
    if not os.path.exists(model_path) or not os.path.exists(exp1_path):
        raise FileNotFoundError("Offline CARE_COM model or Exp1 data not found. Run train_care_predictors.py first.")
        
    xgb = XGBRegressor(**{"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbosity": 0})
    xgb.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiment4'))
    from data_loader import load_raw_features
    
    try:
        df, _ = load_raw_features()
    except Exception as e:
        raise RuntimeError(f"Failed to load CARE features: {e}")
    pairs = []
    for i in range(len(active_experts)):
        for j in range(i+1, len(active_experts)):
            ei = active_experts[i]
            ej = active_experts[j]
            
            # Find in dataframe
            row = df[((df["Expert_A"] == ei) & (df["Expert_B"] == ej)) | ((df["Expert_A"] == ej) & (df["Expert_B"] == ei))]
            if len(row) > 0:
                features = row.iloc[0][['Weight_Distance', 'Weight_Cosine', 'Activation_Similarity', 'Output_Similarity', 'Routing_Similarity', 'Usage_Frequency', 'Jaccard_Overlap', 'Usage_Asymmetry', 'Routing_JSD_Proxy', 'Routing_NPMI_Proxy', 'Specialization_Diff']].values.astype(np.float64)
                scaled = scaler.transform(features.reshape(1, -1))
                pred_damage = xgb.predict(scaled)[0]
                pairs.append((ei, ej, pred_damage))
            else:
                pairs.append((ei, ej, 999.0)) # Fallback if missing
                
    # Sort by smallest predicted damage
    pairs.sort(key=lambda x: x[2])
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, choices=["Random", "Parameter", "Usage", "CARE_COM"])
    parser.add_argument("--mode", type=str, required=True, choices=["one_shot", "iterative"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(f"results/exp5/{args.strategy}_{args.mode}", exist_ok=True)
    
    model, tokenizer = load_base_model()
    engine = MoECompressionEngine(model)
    
    metrics_log = []
    
    # Base Evaluation at 64
    print(f"--- Compression Level 64 ---")
    tier1 = evaluate_tier1(model, tokenizer)
    tier2 = evaluate_tier2(model, tokenizer, args.seed)
    
    metrics_log.append({"level": 64, "tier1": tier1, "tier2": tier2})
    
    # Pre-calculate rankings for one-shot
    if args.mode == "one_shot":
        print(f"Generating One-Shot Ranking for {args.strategy}...")
        if args.strategy == "Random":
            active = engine.active_experts
            pairs = []
            for i in range(len(active)):
                for j in range(i+1, len(active)):
                    pairs.append((active[i], active[j], np.random.rand()))
            pairs.sort(key=lambda x: x[2])
        elif args.strategy == "Parameter":
            pairs = get_parameter_similarity_ranking(engine)
        elif args.strategy == "CARE_COM":
            pairs = get_care_com_ranking(engine, engine.active_experts)
        else:
            raise NotImplementedError(f"Strategy {args.strategy} one_shot not implemented yet.")
            
        master_ranking = pairs
        
    for target in TIER_1_LEVELS[1:]: # Skip 64
        merges_needed = engine.current_num_experts - target
        print(f"--- Compressing from {engine.current_num_experts} to {target} (Needs {merges_needed} merges) ---")
        
        if args.mode == "iterative":
            # Recompute ranking
            if args.strategy == "Parameter":
                current_ranking = get_parameter_similarity_ranking(engine)
            else:
                raise NotImplementedError(f"Strategy {args.strategy} iterative not implemented yet.")
        else:
            # Filter master ranking to active experts
            active_set = set(engine.active_experts)
            current_ranking = [(i, j, s) for (i, j, s) in master_ranking if i in active_set and j in active_set and i != j]
            
        merges_to_apply = greedy_conflict_resolution(current_ranking, merges_needed)
        
        for idx in range(len(merges_to_apply)):
            i, j = merges_to_apply[idx]
            print(f"  Merging expert {j} into {i}")
            mapping = engine.merge_experts(i, j)
            
            # Update the indices for all SUBSEQUENT merges in this step
            for next_idx in range(idx + 1, len(merges_to_apply)):
                next_i, next_j = merges_to_apply[next_idx]
                merges_to_apply[next_idx] = (mapping[next_i], mapping[next_j])
                
            # Update master_ranking for future steps in one-shot mode
            if args.mode == "one_shot":
                for next_idx in range(len(master_ranking)):
                    next_i, next_j, s = master_ranking[next_idx]
                    master_ranking[next_idx] = (mapping[next_i], mapping[next_j], s)
            
        print("Evaluating new state...")
        tier1 = evaluate_tier1(model, tokenizer)
        tier2 = {}
        if target in TIER_2_LEVELS:
            tier2 = evaluate_tier2(model, tokenizer, args.seed)
            
        metrics_log.append({"level": target, "tier1": tier1, "tier2": tier2})
        
        # Save metrics incrementally
        with open(f"results/exp5/{args.strategy}_{args.mode}/metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=4)
            
        # Optional: Save checkpoint
        # model.save_pretrained(f"results/exp5/{args.strategy}_{args.mode}/checkpoint_{target}")
        
    print(f"Finished {args.strategy} ({args.mode}). Results saved.")

if __name__ == "__main__":
    main()
