"""
EXPERIMENT 7A - PHASE 4: NEURON INTERVENTION
==============================================
Runs causal runtime interventions on sampled neurons and
records accuracy and continuous metrics (probability, margin).
"""

import os
import json
import torch
import pandas as pd
import types
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL_ID, HF_REVISION, TARGET_LAYER_IDX,
    DATA_DIR, INTERVENTION_DIR, EXP7A_RESULTS_DIR, DEVICE, DTYPE,
    ensure_dirs, mark_task, is_task_completed
)
from phase2_sensitivity import format_arc_question, compute_choice_logprob, patched_expert_forward

def evaluate_on_dataset(model, tokenizer, dataset, mask_config=None):
    """
    Evaluates the model on the dataset. If mask_config=(expert_id, neuron_idx),
    masks that specific neuron in the patched forward pass.
    """
    experts_module = model.model.layers[TARGET_LAYER_IDX].mlp.experts
    experts_module._mask_config = mask_config
    for expert in experts_module:
        expert._mask_config = mask_config
    
    correct_count = 0
    total = len(dataset)
    
    sum_p_correct = 0.0
    sum_margin = 0.0
    sum_loss = 0.0
    sum_logit_margin = 0.0
    
    print(f"    [Debug] Model device: {next(model.parameters()).device}")
    for idx, item in enumerate(dataset):
        if idx % 1 == 0:
            print(f"    Evaluating item {idx}/{total}...")
            
        prompt, candidates, correct_idx = format_arc_question(item)
        
        log_probs = []
        for choice in candidates:
            log_probs.append(compute_choice_logprob(model, tokenizer, prompt, choice))
            
        log_probs = torch.stack(log_probs)
        probs = torch.nn.functional.softmax(log_probs, dim=0)
        
        pred_idx = torch.argmax(probs).item()
        if pred_idx == correct_idx:
            correct_count += 1
            
        p_correct = probs[correct_idx].item()
        sum_p_correct += p_correct
        
        # Margin: P(correct) - max(P(incorrect))
        incorrect_probs = torch.cat([probs[:correct_idx], probs[correct_idx+1:]])
        max_incorrect_prob = torch.max(incorrect_probs).item() if len(incorrect_probs) > 0 else 0.0
        margin = p_correct - max_incorrect_prob
        sum_margin += margin
        
        # Continuous metrics: Loss and Logit Margin
        correct_log_prob = log_probs[correct_idx].item()
        incorrect_log_probs = torch.cat([log_probs[:correct_idx], log_probs[correct_idx+1:]])
        max_incorrect_log_prob = torch.max(incorrect_log_probs).item() if len(incorrect_log_probs) > 0 else 0.0
        
        loss = -correct_log_prob
        logit_margin = correct_log_prob - max_incorrect_log_prob
        
        sum_loss += loss
        sum_logit_margin += logit_margin
        
    accuracy = correct_count / total
    avg_p_correct = sum_p_correct / total
    avg_margin = sum_margin / total
    avg_loss = sum_loss / total
    avg_logit_margin = sum_logit_margin / total
    
    return accuracy, avg_p_correct, avg_margin, avg_loss, avg_logit_margin

def run_interventions():
    task_id = "phase4_intervention"
    if is_task_completed(task_id):
        print("[Phase 4] Interventions already completed. Skipping.")
        return

    ensure_dirs()
    
    csv_path = os.path.join(EXP7A_RESULTS_DIR, "intervention", "sampled_neurons.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run phase3 first.")
    
    sampled_df = pd.read_csv(csv_path)
    
    ds_path = os.path.join(DATA_DIR, "D_proxy.pt")
    d_proxy = torch.load(ds_path)
    print(f"[Phase 4] Loaded D_proxy ({len(d_proxy)} examples).")
    
    print(f"[Phase 4] Loading model {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        revision=HF_REVISION,
        torch_dtype=DTYPE, 
        device_map=DEVICE, 
        trust_remote_code=True
    )
    model.eval()
    
    # Patch the target layer experts individually
    experts_module = model.model.layers[TARGET_LAYER_IDX].mlp.experts
    experts_module._captured_activations = []
    
    for e_idx, expert in enumerate(experts_module):
        expert._expert_idx = e_idx
        expert._global_storage = None # VERY IMPORTANT: None prevents memory leak in Phase 4!
        expert._mask_config = None
        expert.forward = types.MethodType(patched_expert_forward, expert)
    
    # Baseline
    print("[Phase 4] Evaluating baseline...")
    with torch.no_grad():
        base_acc, base_p_corr, base_margin, base_loss, base_logit_margin = evaluate_on_dataset(model, tokenizer, d_proxy, mask_config=None)
    print(f"  Baseline Accuracy: {base_acc:.4f}, Loss: {base_loss:.4f}, Logit Margin: {base_logit_margin:.4f}")
    
    # Interventions
    results = []
    out_csv = os.path.join(INTERVENTION_DIR, "intervention_results.csv")
    
    start_idx = 0
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict('records')
        start_idx = len(results)
        print(f"[Phase 4] Found {start_idx} existing intervention results. Resuming...")
        
    print(f"[Phase 4] Running {len(sampled_df)} interventions...")
    
    for i, row in sampled_df.iterrows():
        if i < start_idx:
            continue
            
        expert_id = int(row['expert_idx'])
        neuron_id = int(row['neuron_idx'])
        
        print(f"  [{i+1}/{len(sampled_df)}] Masking E{expert_id} N{neuron_id}...")
        
        with torch.no_grad():
            int_acc, int_p_corr, int_margin, int_loss, int_logit_margin = evaluate_on_dataset(
                model, tokenizer, d_proxy, mask_config=(expert_id, neuron_id)
            )
            
        delta_acc = base_acc - int_acc
        delta_p = base_p_corr - int_p_corr
        delta_m = base_margin - int_margin
        delta_loss = int_loss - base_loss # Loss increases when capability drops
        delta_logit_margin = base_logit_margin - int_logit_margin
        
        results.append({
            "expert_idx": expert_id,
            "neuron_idx": neuron_id,
            "stratum": row['stratum'],
            "T_K": row['T_K'],
            "base_acc": base_acc,
            "int_acc": int_acc,
            "base_loss": base_loss,
            "int_loss": int_loss,
            "delta_acc": delta_acc,
            "delta_loss": delta_loss,
            "delta_logit_margin": delta_logit_margin
        })
        
        # Save results incrementally
        pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"[Phase 4] Saved all intervention results to {out_csv}")
    
    print(f"[Phase 4] Saved intervention results to {out_csv}")
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_interventions()
