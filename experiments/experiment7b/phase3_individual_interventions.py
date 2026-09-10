"""
EXPERIMENT 7B - PHASE 3: INDIVIDUAL EXPERT ABLATIONS (STAGE B1)
===============================================================
Measures functional damage when each expert is ablated individually:
D(i) = Damage under ablation of expert i alone.
Ablations are evaluated across both:
1. Token-level KL divergence D_KL(i) on Wikitext-2.
2. Downstream ARC-Challenge loss, margin, and accuracy damage.
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

def run_individual_interventions():
    task_id = "phase3_individual_interventions"
    if is_task_completed(task_id):
        print("[Phase 3] Individual interventions already completed. Skipping.")
        return

    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 3: STAGE B1 — INDIVIDUAL EXPERT ABLATIONS")
    print("=" * 70)

    # 1. Load candidates to identify unique experts
    cand_path = os.path.join(CANDIDATE_DIR, "candidate_pairs.csv")
    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Missing candidates: {cand_path}. Run phase1 first.")
    cand_df = pd.read_csv(cand_path)
    
    unique_experts = sorted(list(set(cand_df["expert_i"].unique()) | set(cand_df["expert_j"].unique())))
    print(f"[Phase 3] Found {len(unique_experts)} unique experts across the 18 candidate pairs:")
    print(f"          {unique_experts}")

    # 2. Check for resuming
    out_csv = os.path.join(INTERVENTION_DIR, "individual_interventions.csv")
    completed_experts = set()
    results = []
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict("records")
        completed_experts = set(existing_df["expert_id"].unique())
        print(f"[Phase 3] Resuming with {len(completed_experts)} already evaluated experts.")

    # 3. Load Model
    model, tokenizer = load_base_model()
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)

    # 4. Prepare Evaluation Datasets
    eval_chunks = prepare_wikitext_eval_batches(tokenizer)
    arc_dataset = load_arc_partition("D_proxy.pt")

    # Load baseline measurements
    baseline_file = os.path.join(MERGES_DIR, "baseline_eval.json")
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            base_stats = json.load(f)
    else:
        print("[Phase 3] Baseline file missing, evaluating fresh baseline...")
        base_arc = evaluate_arc_dataset(model, tokenizer, arc_dataset)
        base_stats = {
            "arc_accuracy": base_arc["accuracy"],
            "arc_avg_loss": base_arc["avg_loss"],
            "arc_avg_margin": base_arc["avg_margin"],
            "arc_avg_logit_margin": base_arc["avg_logit_margin"],
        }
        with open(baseline_file, "w") as f:
            json.dump(base_stats, f, indent=2)

    baseline_logprobs = collect_baseline_wikitext_logits(model, eval_chunks)

    # 5. Evaluate Individual Ablations Loop
    print(f"\n[Phase 3] Evaluating individual ablations for {len(unique_experts)} experts...")
    
    for eid in tqdm(unique_experts, desc="Individual Ablations"):
        if eid in completed_experts:
            continue

        print(f"\n  -> Ablating Expert E{eid} alone...")
        restore_fn = apply_expert_ablation(moe_block, [eid])

        try:
            # Measure damage under ablation of E_eid
            d_kl = evaluate_wikitext_oracle_kl(model, eval_chunks, baseline_logprobs)
            ablated_arc = evaluate_arc_dataset(model, tokenizer, arc_dataset)

            delta_loss = ablated_arc["avg_loss"] - base_stats["arc_avg_loss"]
            delta_margin = base_stats["arc_avg_margin"] - ablated_arc["avg_margin"]
            delta_acc = base_stats["arc_accuracy"] - ablated_arc["accuracy"]

            record = {
                "expert_id": eid,
                "D_KL": d_kl,
                "delta_loss": delta_loss,
                "delta_margin": delta_margin,
                "delta_acc": delta_acc,
                "base_loss": base_stats["arc_avg_loss"],
                "ablated_loss": ablated_arc["avg_loss"],
                "base_acc": base_stats["arc_accuracy"],
                "ablated_acc": ablated_arc["accuracy"]
            }
            results.append(record)
            completed_experts.add(eid)

            print(f"     Ablation E{eid} -> KL Damage: {d_kl:.6f} | Delta Loss: {delta_loss:+.4f} | Delta Acc: {delta_acc:+.4f}")
            pd.DataFrame(results).to_csv(out_csv, index=False)

        finally:
            restore_fn()

    cleanup_vram()
    print(f"\n[Phase 3] Successfully completed individual ablations! Saved to {out_csv}")
    mark_task(task_id, "completed", details=f"Evaluated {len(results)} individual expert ablations.")

if __name__ == "__main__":
    run_individual_interventions()
