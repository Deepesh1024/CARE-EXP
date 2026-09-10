"""
EXPERIMENT 7B - FEATURE LOADER
==============================
Loads the 2,016 middle-layer pairs from results/exp1/output.json
and computes the 4 CARE capability descriptors from Experiment 2
to construct the complete 11-dimensional feature matrix X.
"""

import json
import numpy as np
import pandas as pd
import os

from config import EXP1_OUTPUT_JSON, LOCAL_FEATURES, ORIGINAL_FEATURES

EPSILON = 1e-10

def _compute_per_expert_stats(df: pd.DataFrame) -> dict:
    """Compute per-expert marginal statistics across all 2016 pairs."""
    all_experts = sorted(set(df["Expert_A"].unique()) | set(df["Expert_B"].unique()))
    stats = {}
    for exp_id in all_experts:
        mask = (df["Expert_A"] == exp_id) | (df["Expert_B"] == exp_id)
        pairs = df[mask]
        stats[int(exp_id)] = {
            "Usage_Mean": float(pairs["Usage_Frequency"].mean())
        }
    return stats

def _compute_usage_asymmetry(df: pd.DataFrame, expert_stats: dict) -> np.ndarray:
    vals = np.zeros(len(df), dtype=np.float32)
    for k, (_, row) in enumerate(df.iterrows()):
        ua = expert_stats.get(int(row["Expert_A"]), {}).get("Usage_Mean", 0.0)
        ub = expert_stats.get(int(row["Expert_B"]), {}).get("Usage_Mean", 0.0)
        vals[k] = abs(ua - ub)
    return vals

def _compute_routing_jsd_proxy(df: pd.DataFrame) -> np.ndarray:
    rs = np.clip(df["Routing_Similarity"].values, -1.0, 1.0)
    jo = np.clip(df["Jaccard_Overlap"].values, 0.0, 1.0)
    return ((1.0 - rs) * (1.0 - jo)).astype(np.float32)

def _compute_routing_npmi_proxy(df: pd.DataFrame, expert_stats: dict) -> np.ndarray:
    global_mean = np.mean([v["Usage_Mean"] for v in expert_stats.values()])
    vals = np.zeros(len(df), dtype=np.float32)
    for k, (_, row) in enumerate(df.iterrows()):
        u_i = expert_stats.get(int(row["Expert_A"]), {}).get("Usage_Mean", global_mean)
        u_j = expert_stats.get(int(row["Expert_B"]), {}).get("Usage_Mean", global_mean)
        p_i = u_i / max(global_mean * 3, EPSILON)
        p_j = u_j / max(global_mean * 3, EPSILON)
        p_ij = max(float(row["Jaccard_Overlap"]) * float(row["Usage_Frequency"]), EPSILON)

        pmi = np.log(p_ij / max(p_i * p_j, EPSILON))
        neg_log_pij = -np.log(max(p_ij, EPSILON))

        vals[k] = float(pmi / neg_log_pij) if neg_log_pij > EPSILON else 0.0
    return np.clip(vals, -1.0, 1.0)

def _compute_specialization_diff(df: pd.DataFrame, expert_stats: dict) -> np.ndarray:
    spec = {eid: 1.0 / (v["Usage_Mean"] + EPSILON) for eid, v in expert_stats.items()}
    vals = np.zeros(len(df), dtype=np.float32)
    for k, (_, row) in enumerate(df.iterrows()):
        sa = spec.get(int(row["Expert_A"]), 0.0)
        sb = spec.get(int(row["Expert_B"]), 0.0)
        vals[k] = abs(sa - sb)
    return vals

def load_all_middle_layer_features():
    """
    Loads all 2,016 middle layer pairs (Seq_Len=512) and computes all 11 local features.
    Returns DataFrame with columns: Expert_A, Expert_B, Oracle_KL, and LOCAL_FEATURES.
    """
    if not os.path.exists(EXP1_OUTPUT_JSON):
        raise FileNotFoundError(f"Missing {EXP1_OUTPUT_JSON}")
        
    with open(EXP1_OUTPUT_JSON, "r") as f:
        data = json.load(f)
        
    records = [
        r for r in data.get("results", [])
        if r.get("Layer") == "middle" and r.get("Seq_Len") == 512
    ]
    if len(records) != 2016:
        raise ValueError(f"Expected 2016 middle layer records, found {len(records)}")
        
    df = pd.DataFrame(records)
    
    # Sort deterministically by (Expert_A, Expert_B)
    df = df.sort_values(["Expert_A", "Expert_B"]).reset_index(drop=True)
    
    # Compute per-expert stats
    expert_stats = _compute_per_expert_stats(df)
    
    # Add the 4 CARE descriptors
    df["Usage_Asymmetry"] = _compute_usage_asymmetry(df, expert_stats)
    df["Routing_JSD_Proxy"] = _compute_routing_jsd_proxy(df)
    df["Routing_NPMI_Proxy"] = _compute_routing_npmi_proxy(df, expert_stats)
    df["Specialization_Diff"] = _compute_specialization_diff(df, expert_stats)
    
    return df
