import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from config import BASE_MODEL_ID, HF_REVISION, TARGET_LAYER_IDX, DEVICE, DTYPE, _PROJECT_ROOT, is_task_completed, mark_task
from phase4_intervention import evaluate_on_dataset

def run_smoke_test():
    task_id = "smoke_test_regression"
    if is_task_completed(task_id):
        print("[Smoke Test] Regression test already passed. Skipping.")
        return
        
    print(f"Loading from Device: {DEVICE}")
    exp7a5_csv = os.path.join(_PROJECT_ROOT, "results", "exp7a5", "intervention", "intervention_results.csv")
    df = pd.read_csv(exp7a5_csv)
    
    # Pick the first neuron
    row = df.iloc[0]
    expert_id = int(row['expert_idx'])
    neuron_id = int(row['neuron_idx'])
    old_delta_margin = float(row['delta_logit_margin'])
    old_tk = float(row['T_K'])
    
    print(f"Testing Neuron: Layer {TARGET_LAYER_IDX}, Expert {expert_id}, Neuron {neuron_id}")
    
    # Load dataset
    ds_path = os.path.join(_PROJECT_ROOT, "results", "exp7a", "data", "D_proxy.pt")
    d_proxy = torch.load(ds_path)
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        revision=HF_REVISION,
        torch_dtype=DTYPE, 
        device_map=DEVICE, 
        trust_remote_code=True
    )
    model.eval()
    
    # Patch experts
    from phase2_sensitivity import patched_expert_forward
    import types
    experts_module = model.model.layers[TARGET_LAYER_IDX].mlp.experts
    for e_idx, expert in enumerate(experts_module):
        expert._expert_idx = e_idx
        expert._global_storage = None
        expert._mask_config = None
        expert.forward = types.MethodType(patched_expert_forward, expert)
        
    print("Running baseline...")
    with torch.no_grad():
        b_acc, b_p, b_margin, b_loss, new_base_margin = evaluate_on_dataset(model, tokenizer, d_proxy, mask_config=None)
        
    print("Running intervention...")
    with torch.no_grad():
        i_acc, i_p, i_margin, i_loss, new_int_margin = evaluate_on_dataset(model, tokenizer, d_proxy, mask_config=(expert_id, neuron_id))
        
    new_delta_margin = new_base_margin - new_int_margin
    
    print("\n--- RESULTS ---")
    print(f"Old Delta Margin: {old_delta_margin}")
    print(f"New Delta Margin: {new_delta_margin} (Diff: {abs(new_delta_margin - old_delta_margin):.8e})")
    
    # Assert tolerance (relaxed for bfloat16 non-determinism)
    diff = abs(new_delta_margin - old_delta_margin)
    if diff > 1e-3:
        raise ValueError(f"Regression Test Failed! Diff {diff} is too large for bfloat16 tolerance.")
        
    print(f"[Smoke Test] Regression test passed successfully! (Difference {diff:.8e} is within bfloat16 tolerance)")
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_smoke_test()
