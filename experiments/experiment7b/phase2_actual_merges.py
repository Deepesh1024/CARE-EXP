"""
EXPERIMENT 7B - PHASE 2: ACTUAL EXPERT MERGES (STAGE A)
=======================================================
Performs in-place parameter averaging for each selected candidate pair.
Measures actual post-merge functional damage:
1. D_actual^KL: Token-level Oracle KL divergence on Wikitext-2 validation.
2. D_actual^task: ARC-Challenge loss degradation, margin loss, and accuracy drop.
Computes prediction error E(i, j) = D_actual^KL(i, j) - D_pred(i, j).
"""

import os
import sys
import json
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ensure_dirs, mark_task, is_task_completed,
    CANDIDATE_DIR, MERGES_DIR, TARGET_LAYER_IDX,
    DEVICE, DTYPE, RANDOM_SEED
)
from utils.model_utils import (
    load_base_model, get_target_moe_block,
    apply_in_place_merge, cleanup_vram
)
from utils.evaluation import (
    prepare_wikitext_eval_batches,
    collect_baseline_wikitext_logits,
    evaluate_wikitext_oracle_kl,
    load_arc_partition,
    evaluate_arc_dataset
)

def run_actual_merges():
    task_id = "phase2_actual_merges"
    if is_task_completed(task_id):
        print("[Phase 2] Actual merges already completed. Skipping.")
        return

    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 2: STAGE A — ACTUAL MERGE DAMAGE EVALUATION")
    print("=" * 70)

    # 1. Load candidates
    cand_path = os.path.join(CANDIDATE_DIR, "candidate_pairs.csv")
    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Missing candidates file: {cand_path}. Run phase1 first.")
    cand_df = pd.read_csv(cand_path)
    print(f"[Phase 2] Loaded {len(cand_df)} candidate pairs.")

    # 2. Check for resuming
    out_csv = os.path.join(MERGES_DIR, "actual_merge_results.csv")
    completed_pairs = set()
    results = []
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict("records")
        completed_pairs = set(existing_df["pair_id"].unique())
        print(f"[Phase 2] Resuming with {len(completed_pairs)} already evaluated pairs.")

    # 3. Load Model
    model, tokenizer = load_base_model()
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)

    # 4. Prepare Evaluation Datasets
    print("\n[Phase 2] Initializing evaluation datasets...")
    eval_chunks = prepare_wikitext_eval_batches(tokenizer)
    arc_dataset = load_arc_partition("D_proxy.pt")

    # Precompute / Load baseline measurements
    baseline_file = os.path.join(MERGES_DIR, "baseline_eval.json")
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            base_stats = json.load(f)
        print(f"[Phase 2] Loaded existing baseline stats: {base_stats}")
    else:
        print("[Phase 2] Evaluating unmerged baseline model...")
        base_arc = evaluate_arc_dataset(model, tokenizer, arc_dataset)
        base_stats = {
            "arc_accuracy": base_arc["accuracy"],
            "arc_avg_loss": base_arc["avg_loss"],
            "arc_avg_margin": base_arc["avg_margin"],
            "arc_avg_logit_margin": base_arc["avg_logit_margin"],
        }
        with open(baseline_file, "w") as f:
            json.dump(base_stats, f, indent=2)
        print(f"[Phase 2] Baseline established: {base_stats}")

    baseline_logprobs = collect_baseline_wikitext_logits(model, eval_chunks)

    # 5. Evaluate Merges Loop
    print(f"\n[Phase 2] Executing in-place merges for {len(cand_df)} candidate pairs...")
    
    for _, row in tqdm(cand_df.iterrows(), total=len(cand_df), desc="Merge Evaluations"):
        pair_id = row["pair_id"]
        if pair_id in completed_pairs:
            continue

        ei = int(row["expert_i"])
        ej = int(row["expert_j"])
        d_pred = float(row["D_pred"])

        print(f"\n  -> Evaluating Pair {pair_id}: E{ei} <-> E{ej} ({row['selection_stratum']}, rank {row['selection_rank']})...")

        # Perform in-place merge
        restore_fn = apply_in_place_merge(moe_block, ei, ej)

        try:
            # Evaluate In-Domain Oracle KL
            d_actual_kl = evaluate_wikitext_oracle_kl(model, eval_chunks, baseline_logprobs)

            # Evaluate Downstream ARC Capability
            merged_arc = evaluate_arc_dataset(model, tokenizer, arc_dataset)
            delta_loss = merged_arc["avg_loss"] - base_stats["arc_avg_loss"]
            delta_margin = base_stats["arc_avg_margin"] - merged_arc["avg_margin"]
            delta_acc = base_stats["arc_accuracy"] - merged_arc["accuracy"]

            # Compute Genuine Prediction Error (Wikitext KL units)
            error_kl = d_actual_kl - d_pred

            record = {
                "pair_id": pair_id,
                "expert_i": ei,
                "expert_j": ej,
                "selection_stratum": row["selection_stratum"],
                "selection_rank": row["selection_rank"],
                "D_pred": d_pred,
                "D_actual_KL": d_actual_kl,
                "Error_KL": error_kl,
                "delta_loss": delta_loss,
                "delta_margin": delta_margin,
                "delta_acc": delta_acc,
                "base_acc": base_stats["arc_accuracy"],
                "merged_acc": merged_arc["accuracy"],
                "base_loss": base_stats["arc_avg_loss"],
                "merged_loss": merged_arc["avg_loss"]
            }
            results.append(record)
            completed_pairs.add(pair_id)

            print(f"     Actual KL: {d_actual_kl:.6f} | Pred KL: {d_pred:.6f} | Error: {error_kl:+.6f}")
            print(f"     Delta Loss: {delta_loss:+.4f} | Delta Margin: {delta_margin:+.4f} | Delta Acc: {delta_acc:+.4f}")

            # Save incrementally
            pd.DataFrame(results).to_csv(out_csv, index=False)

        finally:
            # Crucial: Always restore baseline parameters cleanly
            restore_fn()

    cleanup_vram()
    print(f"\n[Phase 2] Successfully completed all merges! Results written to {out_csv}")
    mark_task(task_id, "completed", details=f"Evaluated {len(results)} merged pairs.")

if __name__ == "__main__":
    run_actual_merges()
