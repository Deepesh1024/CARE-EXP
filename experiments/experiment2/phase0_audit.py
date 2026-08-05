"""
CARE-MoE Experiment 2 — Phase 0: Oracle Feature Audit
=======================================================
Classify every column in output.json into:
  - Pre-Merge (CARE-Eligible)
  - Oracle-Dependent (CARE-Rejected)
  - Metadata (not a feature)
  - Target

Produces:
  results/exp2/tables/oracle_audit.csv
  stdout: formatted audit table
"""

import pandas as pd

from config import (
    ORIGINAL_FEATURES,
    EXCLUDE_COLS,
    TARGET,
    TABLES_DIR,
)
from utils import set_global_seed, ensure_dirs, load_raw_data, save_csv


# ──────────────────────────────────────────────
# Feature Classification Registry
# ──────────────────────────────────────────────
# Every column in output.json must appear here.
AUDIT_REGISTRY = {
    # --- Pre-Merge Features (CARE-Eligible) ---
    "Weight_Distance": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "L2 norm between flattened expert weight tensors. "
                  "Computable directly from model parameters without any forward pass.",
    },
    "Weight_Cosine": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "Cosine similarity between flattened expert weight tensors. "
                  "Computable directly from model parameters without any forward pass.",
    },
    "Activation_Similarity": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "Mean cosine similarity of intermediate activations between two experts "
                  "on calibration data. Requires only a single forward pass through the "
                  "original (unmerged) model.",
    },
    "Output_Similarity": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "Mean cosine similarity of expert output tensors on calibration data. "
                  "Requires only a single forward pass through the original model.",
    },
    "Routing_Similarity": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "Pearson correlation between router probability vectors for the two experts. "
                  "Computed from the original model's gating network output.",
    },
    "Usage_Frequency": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "Fraction of tokens routed to either expert (|A∪B|/N). "
                  "Computed from the original router's top-k selection.",
    },
    "Jaccard_Overlap": {
        "classification": "Pre-Merge",
        "care_eligible": True,
        "reason": "Jaccard index of token sets routed to each expert (|A∩B|/|A∪B|). "
                  "Computed from the original router's top-k selection.",
    },

    # --- Oracle-Dependent Features (CARE-Rejected) ---
    "CrossEntropy_Delta": {
        "classification": "Oracle-Dependent",
        "care_eligible": False,
        "reason": "Difference in cross-entropy loss between original and merged model. "
                  "REQUIRES creating the merged expert and running a forward pass through "
                  "the merged model. Cannot be computed pre-merge.",
    },
    "Hidden_L2_Drift": {
        "classification": "Oracle-Dependent",
        "care_eligible": False,
        "reason": "L2 distance between hidden states of original vs. merged model at the "
                  "MoE layer. REQUIRES creating the merged expert and running both original "
                  "and merged forward passes.",
    },
    "Router_Entropy_Orig": {
        "classification": "Oracle-Dependent",
        "care_eligible": False,
        "reason": "Mean entropy of the original router's softmax distribution. Although this "
                  "measures the UNmerged router, it is computed inside the oracle evaluation "
                  "function (run_oracle_pair) as a byproduct. Empirically verified to be "
                  "CONSTANT across all pairs within a layer (std=0.0), making it useless as "
                  "a pairwise feature regardless of classification. Extracting it independently "
                  "would require a separate calibration run not present in the current pipeline.",
    },
    "Router_Entropy_Merged": {
        "classification": "Oracle-Dependent",
        "care_eligible": False,
        "reason": "Mean entropy of the merged model's router distribution. "
                  "REQUIRES creating the merged expert and running a forward pass.",
    },
    "Top1_Routing_Agreement": {
        "classification": "Oracle-Dependent",
        "care_eligible": False,
        "reason": "Fraction of tokens where the original and merged router agree on top-1 "
                  "expert selection. REQUIRES the merged model.",
    },
    "TopK_Routing_Agreement": {
        "classification": "Oracle-Dependent",
        "care_eligible": False,
        "reason": "Fraction of tokens where the original and merged router agree on top-k "
                  "expert selection. REQUIRES the merged model.",
    },

    # --- Target ---
    "Oracle_KL": {
        "classification": "Target",
        "care_eligible": False,
        "reason": "KL divergence between original and merged model output distributions. "
                  "This IS the oracle — the quantity we are trying to predict.",
    },
    "Oracle_KL_SplitA": {
        "classification": "Target (Split)",
        "care_eligible": False,
        "reason": "Oracle KL computed on first half of calibration data. "
                  "Used for split-sample stability validation only.",
    },
    "Oracle_KL_SplitB": {
        "classification": "Target (Split)",
        "care_eligible": False,
        "reason": "Oracle KL computed on second half of calibration data. "
                  "Used for split-sample stability validation only.",
    },

    # --- Metadata ---
    "Seq_Len": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Calibration sequence length. Experimental condition, not a feature.",
    },
    "Layer": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Network layer label (first/middle/last). Used to derive Relative_Depth.",
    },
    "Expert_A": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Index of first expert in the pair. Identity information, not a feature.",
    },
    "Expert_B": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Index of second expert in the pair. Identity information, not a feature.",
    },
    "Random_Baseline": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Random number for baseline comparison. Not a meaningful feature.",
    },
    "Runtime_Sec": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Wall-clock time for oracle evaluation. Infrastructure metric.",
    },
    "Max_VRAM_MB": {
        "classification": "Metadata",
        "care_eligible": False,
        "reason": "Peak GPU memory during oracle evaluation. Infrastructure metric.",
    },
}


def run_audit() -> pd.DataFrame:
    """Execute the Oracle Feature Audit.

    Returns
    -------
    pd.DataFrame with columns: Column, Classification, CARE_Eligible, Reason
    """
    # Load raw data to verify all columns are covered
    raw_df = load_raw_data()
    raw_columns = set(raw_df.columns)

    # Build audit table
    rows = []
    for col in sorted(raw_columns):
        entry = AUDIT_REGISTRY.get(col, {
            "classification": "UNKNOWN",
            "care_eligible": False,
            "reason": "Column not found in audit registry — requires manual classification.",
        })
        rows.append({
            "Column": col,
            "Classification": entry["classification"],
            "CARE_Eligible": "✓" if entry["care_eligible"] else "✗",
            "Reason": entry["reason"],
        })

    audit_df = pd.DataFrame(rows)

    # Validation: check for unregistered columns
    registered = set(AUDIT_REGISTRY.keys())
    unregistered = raw_columns - registered
    if unregistered:
        print(f"[Phase 0] WARNING: Unregistered columns found: {unregistered}")

    missing = registered - raw_columns
    if missing:
        print(f"[Phase 0] NOTE: Registry contains columns not in data: {missing}")

    return audit_df


def main():
    set_global_seed()
    ensure_dirs()

    print("=" * 70)
    print("PHASE 0 — ORACLE FEATURE AUDIT")
    print("=" * 70)

    audit_df = run_audit()

    # Print formatted table
    print("\n" + audit_df.to_markdown(index=False))

    # Summary statistics
    print("\n--- Audit Summary ---")
    counts = audit_df["Classification"].value_counts()
    for cls, count in counts.items():
        print(f"  {cls}: {count}")

    eligible = audit_df[audit_df["CARE_Eligible"] == "✓"]
    print(f"\n  Total CARE-Eligible features: {len(eligible)}")
    print(f"  Eligible columns: {eligible['Column'].tolist()}")

    # Save
    import os
    path = os.path.join(TABLES_DIR, "oracle_audit.csv")
    save_csv(audit_df, path)

    print("\n[Phase 0] Oracle Feature Audit complete.")
    return audit_df


if __name__ == "__main__":
    main()
