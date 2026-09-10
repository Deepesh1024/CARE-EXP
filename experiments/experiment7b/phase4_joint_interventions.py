"""
EXPERIMENT 7B - PHASE 4: JOINT EXPERT ABLATIONS & INTERACTION (STAGE B2)
========================================================================
Measures functional damage under joint ablation of expert pair (i, j):
D(i, j) = Damage under simultaneous ablation of both experts i and j.
Computes non-additive joint functional interaction:
I(i, j) = D(i, j) - D(i) - D(j)
Preserves raw signed interaction values across both:
1. Token-level KL Divergence (I_KL).
2. Downstream ARC-Challenge Loss (I_loss), Margin (I_margin), and Acc (I_acc).
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
    CANDIDATE_DIR, INTERVENTION_DIR, MERGES_DIR,
    TARGET_LAYER_IDX, DEVICE
)
from utils.model_utils import (
    load_base_model, get_target_moe_block,
    apply_expert_ablation, cleanup_vram
)
from utils.evaluation import (
    prepare_wikitext_eval_batches,
    collect_baseline_wikitext_logits,
    evaluate_wikitext_oracle_kl,
    load_arc_partition,
    evaluate_arc_dataset
)

def run_joint_interventions():
    task_id = "phase4_joint_interventions"
    if is_task_completed(task_id):
        print("[Phase 4] Joint interventions already completed. Skipping.")
        return

    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 4: STAGE B2 — JOINT EXPERT ABLATIONS & INTERACTION")
    print("=" * 70)

    # 1. Load candidate pairs
    cand_path = os.path.join(CANDIDATE_DIR, "candidate_pairs.csv")
    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Missing candidates: {cand_path}. Run phase1 first.")
    cand_df = pd.read_csv(cand_path)

    # 2. Load individual intervention results (Phase 3 output)
    indiv_path = os.path.join(INTERVENTION_DIR, "individual_interventions.csv")
    if not os.path.exists(indiv_path):
        raise FileNotFoundError(f"Missing individual ablations: {indiv_path}. Run phase3 first.")
    indiv_df = pd.read_csv(indiv_path)
    indiv_map = {int(r["expert_id"]): r for _, r in indiv_df.iterrows()}

    # Verify all candidate experts are in indiv_map
    for _, row in cand_df.iterrows():
        ei, ej = int(row["expert_i"]), int(row["expert_j"])
        if ei not in indiv_map or ej not in indiv_map:
            raise ValueError(f"Missing individual ablation for E{ei} or E{ej}. Check phase3.")

    # 3. Check for resuming
    out_csv = os.path.join(INTERVENTION_DIR, "joint_interventions.csv")
    completed_pairs = set()
    results = []
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict("records")
        completed_pairs = set(existing_df["pair_id"].unique())
        print(f"[Phase 4] Resuming with {len(completed_pairs)} already evaluated joint pairs.")

    # 4. Load Model
    model, tokenizer = load_base_model()
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)

    # 5. Prepare Evaluation Datasets
    eval_chunks = prepare_wikitext_eval_batches(tokenizer)
    arc_dataset = load_arc_partition("D_proxy.pt")

    # Load baseline measurements
    baseline_file = os.path.join(MERGES_DIR, "baseline_eval.json")
    with open(baseline_file, "r") as f:
        base_stats = json.load(f)

    baseline_logprobs = collect_baseline_wikitext_logits(model, eval_chunks)

    # 6. Evaluate Joint Ablations Loop
    print(f"\n[Phase 4] Evaluating joint ablations for {len(cand_df)} pairs...")
    
    for _, row in tqdm(cand_df.iterrows(), total=len(cand_df), desc="Joint Ablations"):
        pair_id = row["pair_id"]
        if pair_id in completed_pairs:
            continue

        ei = int(row["expert_i"])
        ej = int(row["expert_j"])

        print(f"\n  -> Jointly Ablating Experts E{ei} and E{ej}...")
        restore_fn = apply_expert_ablation(moe_block, [ei, ej])

        try:
            # Measure joint damage D(i, j)
            d_joint_kl = evaluate_wikitext_oracle_kl(model, eval_chunks, baseline_logprobs)
            joint_arc = evaluate_arc_dataset(model, tokenizer, arc_dataset)

            delta_joint_loss = joint_arc["avg_loss"] - base_stats["arc_avg_loss"]
            delta_joint_margin = base_stats["arc_avg_margin"] - joint_arc["avg_margin"]
            delta_joint_acc = base_stats["arc_accuracy"] - joint_arc["accuracy"]

            # Individual damages
            d_i_kl = float(indiv_map[ei]["D_KL"])
            d_j_kl = float(indiv_map[ej]["D_KL"])
            d_i_loss = float(indiv_map[ei]["delta_loss"])
            d_j_loss = float(indiv_map[ej]["delta_loss"])
            d_i_acc = float(indiv_map[ei]["delta_acc"])
            d_j_acc = float(indiv_map[ej]["delta_acc"])

            # Compute interaction quantities: I(i, j) = D(i, j) - D(i) - D(j)
            i_kl = d_joint_kl - d_i_kl - d_j_kl
            i_loss = delta_joint_loss - d_i_loss - d_j_loss
            i_acc = delta_joint_acc - d_i_acc - d_j_acc

            record = {
                "pair_id": pair_id,
                "expert_i": ei,
                "expert_j": ej,
                "selection_stratum": row["selection_stratum"],
                "selection_rank": row["selection_rank"],
                "D_pred": float(row["D_pred"]),
                # Individual damages
                "D_i_KL": d_i_kl,
                "D_j_KL": d_j_kl,
                "D_sum_KL": d_i_kl + d_j_kl,
                "D_i_loss": d_i_loss,
                "D_j_loss": d_j_loss,
                # Joint damage
                "D_joint_KL": d_joint_kl,
                "delta_joint_loss": delta_joint_loss,
                "delta_joint_margin": delta_joint_margin,
                "delta_joint_acc": delta_joint_acc,
                # Non-additive interaction
                "I_KL": i_kl,
                "I_loss": i_loss,
                "I_acc": i_acc
            }
            results.append(record)
            completed_pairs.add(pair_id)

            print(f"     Joint KL: {d_joint_kl:.6f} (D_i={d_i_kl:.6f}, D_j={d_j_kl:.6f}) -> Interaction I_KL: {i_kl:+.6f}")
            print(f"     Joint Loss: {delta_joint_loss:+.4f} -> Interaction I_loss: {i_loss:+.4f}")
            pd.DataFrame(results).to_csv(out_csv, index=False)

        finally:
            restore_fn()

    cleanup_vram()
    print(f"\n[Phase 4] Successfully completed joint ablations! Saved to {out_csv}")
    mark_task(task_id, "completed", details=f"Evaluated {len(results)} joint expert pairs.")

if __name__ == "__main__":
    run_joint_interventions()
