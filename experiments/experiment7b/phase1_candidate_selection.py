"""
EXPERIMENT 7B - PHASE 1: CANDIDATE PAIR SELECTION
=================================================
Scores all 2,016 expert pairs in central Layer 8 using the validated CARE-COM predictor.
Constructs a balanced stratified sample of 18 candidate pairs:
- Group A: Safe / Attractive (Top 15% lowest predicted damage)
- Group B: Moderate / Neutral (40%–60% percentile)
- Group C: Unsafe / Dissimilar (Bottom 15% highest predicted damage)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ensure_dirs, mark_task, is_task_completed,
    EXP1_OUTPUT_JSON, EXP5_MODEL_PATH, EXP5_SCALER_PATH,
    CANDIDATE_DIR, RANDOM_SEED, N_PER_STRATUM, STRATA,
    LOCAL_FEATURES
)

def run_candidate_selection():
    task_id = "phase1_candidate_selection"
    if is_task_completed(task_id):
        print("[Phase 1] Candidate selection already completed. Skipping.")
        return
        
    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 1: STRATIFIED CANDIDATE SELECTION")
    print("=" * 70)
    
    # 1. Load Exp 1 features and compute all 11 CARE descriptors
    print(f"\n[Phase 1] Loading 2,016 pairs and computing 11 CARE features...")
    from utils.feature_loader import load_all_middle_layer_features
    df = load_all_middle_layer_features()
    print(f"Loaded {len(df)} pairs with all {len(LOCAL_FEATURES)} features.")
    
    # 2. Score with CARE-COM model
    print("\n[Phase 1] Scoring candidate pairs with CARE-COM predictor...")
    scaler = joblib.load(EXP5_SCALER_PATH)
    xgb = XGBRegressor()
    xgb.load_model(EXP5_MODEL_PATH)
    
    X = df[LOCAL_FEATURES].values.astype(np.float64)
    X_scaled = scaler.transform(X)
    d_pred = xgb.predict(X_scaled)
    df["D_pred"] = d_pred
    
    # Sort by predicted damage (smallest first: rank 1 = safest)
    df = df.sort_values("D_pred", ascending=True).reset_index(drop=True)
    df["selection_rank"] = df.index + 1
    
    # 3. Stratified Sampling
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_rows = []
    
    total_n = len(df)
    print(f"\n[Phase 1] Performing stratified sampling (Seed = {RANDOM_SEED})...")
    
    for stratum_key, cfg in STRATA.items():
        idx_start = int(cfg["pct_min"] * total_n)
        idx_end = int(cfg["pct_max"] * total_n)
        stratum_pool = df.iloc[idx_start:idx_end].copy()
        
        print(f"  {stratum_key} ({cfg['name']}): Pool size {len(stratum_pool)} (ranks {idx_start+1} to {idx_end})")
        
        # Sample N_PER_STRATUM uniformly without replacement
        sample_indices = rng.choice(len(stratum_pool), size=N_PER_STRATUM, replace=False)
        selected = stratum_pool.iloc[sample_indices].copy()
        selected["selection_stratum"] = stratum_key
        selected["stratum_name"] = cfg["name"]
        sampled_rows.append(selected)
        
    candidates_df = pd.concat(sampled_rows).sort_values("selection_rank").reset_index(drop=True)
    candidates_df["pair_id"] = [f"pair_{i+1:02d}" for i in range(len(candidates_df))]
    
    # Retain required and diagnostic fields
    out_cols = [
        "pair_id", "Expert_A", "Expert_B", "D_pred", "Weight_Distance",
        "Weight_Cosine", "Routing_Similarity", "selection_stratum",
        "stratum_name", "selection_rank", "Oracle_KL"
    ]
    candidates_df = candidates_df.rename(columns={"Expert_A": "expert_i", "Expert_B": "expert_j"})
    out_cols = [c if c not in ["Expert_A", "Expert_B"] else "expert_i" if c == "Expert_A" else "expert_j" for c in out_cols]
    final_df = candidates_df[out_cols]
    
    # Save CSV and JSON
    csv_path = os.path.join(CANDIDATE_DIR, "candidate_pairs.csv")
    json_path = os.path.join(CANDIDATE_DIR, "candidate_pairs.json")
    final_df.to_csv(csv_path, index=False)
    final_df.to_json(json_path, orient="records", indent=2)
    
    print("\n[Phase 1] Selected 18 candidate pairs:")
    print(final_df[["pair_id", "expert_i", "expert_j", "D_pred", "selection_stratum", "selection_rank"]].to_string(index=False))
    print(f"\nSaved candidates to:\n  {csv_path}\n  {json_path}")
    
    mark_task(task_id, "completed", details=f"Sampled {len(final_df)} pairs across 3 strata.")

if __name__ == "__main__":
    run_candidate_selection()
