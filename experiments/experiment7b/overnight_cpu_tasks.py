import os
import sys
import json
import pandas as pd
import numpy as np

def run_stage4():
    print("Running Stage 4: Generate D_pred for all 18 pairs...")
    candidates_path = "../../results/exp7b/candidates/candidate_pairs.csv"
    if not os.path.exists(candidates_path):
        print("Candidate pairs CSV not found!")
        return False
        
    df = pd.read_csv(candidates_path)
    
    # Needs: pair, expert_i, expert_j, risk_bucket, D_pred, rank, all relevant predictor features
    # df already has pair_id, expert_i, expert_j, D_pred, stratum_name (risk_bucket), selection_rank
    
    # We load the full feature set to get all relevant predictor features
    features_path = "../../results/exp1/output.json"
    with open(features_path, 'r') as f:
        exp1 = json.load(f)["results"]
        
    # Build feature lookup
    feat_lookup = {}
    for r in exp1:
        if r["Layer"] == "middle" and r["Seq_Len"] == 512:
            feat_lookup[(int(r["Expert_A"]), int(r["Expert_B"]))] = r
            
    out_rows = []
    for _, row in df.iterrows():
        i, j = int(row["expert_i"]), int(row["expert_j"])
        feats = feat_lookup.get((i, j), {})
        
        row_data = {
            "pair": row["pair_id"],
            "expert_i": i,
            "expert_j": j,
            "risk_bucket": row["stratum_name"],
            "D_pred": row["D_pred"],
            "rank": row["selection_rank"]
        }
        
        local_features = [
            "Random_Baseline", "Weight_Distance", "Weight_Cosine", "Activation_Similarity",
            "Output_Similarity", "Routing_Similarity", "Usage_Frequency", "Jaccard_Overlap",
            "CrossEntropy_Delta", "Hidden_L2_Drift"
        ]
        
        for lf in local_features:
            row_data[lf] = feats.get(lf, np.nan)
            
        out_rows.append(row_data)
        
    out_df = pd.DataFrame(out_rows)
    out_path = "../../results/exp7b/predictions_18_pairs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Stage 4 complete. Saved to {out_path}")
    return True


def run_stage7():
    print("Running Stage 7: Phase 5 Dry Run...")
    candidates_path = "../../results/exp7b/candidates/candidate_pairs.csv"
    df = pd.read_csv(candidates_path)
    
    dry_run_dir = "../../results/exp7b/dry_run"
    os.makedirs(dry_run_dir, exist_ok=True)
    
    mock_actual_path = os.path.join(dry_run_dir, "mock_actual_merge_results.csv")
    mock_joint_path = os.path.join(dry_run_dir, "mock_joint_interventions.csv")
    
    # Generate mock_actual_merge_results.csv
    # Needs columns expected by phase5_analysis.py
    # pair_id, expert_i, expert_j, D_pred, D_actual_KL, ARC_Accuracy, Error_KL
    mock_actual = []
    for _, row in df.iterrows():
        d_pred = row["D_pred"]
        d_actual = d_pred + np.random.normal(0, 0.001)
        mock_actual.append({
            "pair_id": row["pair_id"],
            "expert_i": row["expert_i"],
            "expert_j": row["expert_j"],
            "D_pred": d_pred,
            "D_actual_KL": d_actual,
            "ARC_Accuracy": 0.5 + np.random.normal(0, 0.05),
            "Error_KL": d_actual - d_pred
        })
    pd.DataFrame(mock_actual).to_csv(mock_actual_path, index=False)
    
    # Generate mock_joint_interventions.csv
    mock_joint = []
    for _, row in df.iterrows():
        mock_joint.append({
            "pair_id": row["pair_id"],
            "expert_i": row["expert_i"],
            "expert_j": row["expert_j"],
            "ARC_Accuracy": 0.5 + np.random.normal(0, 0.05),
            "ARC_Accuracy_Delta": np.random.normal(0, 0.02)
        })
    pd.DataFrame(mock_joint).to_csv(mock_joint_path, index=False)
    
    # To run Phase 5 without it crashing looking for real files, we'll patch config temporarily
    # but phase5_analysis.py uses hardcoded paths from config.
    # I will just invoke phase5_analysis.py as a subprocess with DRY_RUN env var,
    # but I need to modify phase5_analysis.py to respect it. 
    # Actually, phase5_analysis.py doesn't have a --dry-run flag.
    # Let's read phase5_analysis.py to see how to inject our mock data.
    return True

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_stage4()
    run_stage7()
