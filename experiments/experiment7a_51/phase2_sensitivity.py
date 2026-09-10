"""
EXPERIMENT 7A - PHASE 2: SENSITIVITY COMPUTATION
=================================================
Computes capability-conditioned Taylor sensitivity (T_K) and 
baselines (Activation magnitude A_n, Weight magnitude W_n) 
for all neurons in the central layer.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import types
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL_ID, HF_REVISION, TARGET_LAYER_IDX,
    DATA_DIR, SENSITIVITY_DIR,
    N_EXPERTS, INTERMEDIATE_SIZE, DEVICE, DTYPE,
    ensure_dirs, mark_task, is_task_completed
)

# ══════════════════════════════════════════════════════════
# Monkey Patching
# ══════════════════════════════════════════════════════════

def patched_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward for a single OlmoeExpert to capture activations and gradients natively.
    """
    if hasattr(self, 'gate_up_proj'):
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        
    if hasattr(self, 'act_fn'):
        act = self.act_fn(gate)
    else:
        act = torch.nn.functional.silu(gate)
        
    current_hidden_states = act * up
    
    # Enable gradient tracking natively
    if current_hidden_states.requires_grad:
        current_hidden_states.retain_grad()
    else:
        current_hidden_states.requires_grad_(True)
        current_hidden_states.retain_grad()
        
    # Masking (Phase 4 support)
    mask_config = getattr(self, "_mask_config", None)
    if mask_config is not None:
        target_expert, target_neuron = mask_config
        if self._expert_idx == target_expert:
            current_hidden_states = current_hidden_states.clone() # avoid inplace
            current_hidden_states[:, target_neuron] = 0.0
            
    # Store for backward pass analysis
    if getattr(self, "_global_storage", None) is not None:
        self._global_storage.append({
            "expert_idx": self._expert_idx,
            "tensor": current_hidden_states,
            "num_tokens": current_hidden_states.size(0)
        })
    
    down = self.down_proj(current_hidden_states)
    return down

# ══════════════════════════════════════════════════════════
# Metric Aggregation
# ══════════════════════════════════════════════════════════

def format_arc_question(item):
    """Format question into context and candidate answers."""
    question = item["question"]
    choices = item["choices"]
    
    # We want to score: "Question: {q}\nAnswer: {c}"
    prompt = f"Question: {question}\nAnswer:"
    
    candidates = []
    correct_idx = -1
    for i, (label, text) in enumerate(zip(choices["label"], choices["text"])):
        candidates.append(f" {text}")
        if label == item["answerKey"]:
            correct_idx = i
            
    return prompt, candidates, correct_idx

def compute_choice_logprob(model, tokenizer, prompt, choice):
    """Compute the log probability of a choice given the prompt."""
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    choice_tokens = tokenizer(choice, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    
    input_ids = torch.cat([prompt_tokens, choice_tokens], dim=1)
    
    outputs = model(input_ids)
    logits = outputs.logits[0, :-1, :] # Shifted
    target_ids = input_ids[0, 1:]
    
    # We only care about the logprob of the choice tokens
    choice_start = prompt_tokens.size(1) - 1
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    choice_log_probs = []
    
    for i in range(choice_start, len(target_ids)):
        choice_log_probs.append(log_probs[i, target_ids[i]])
        
    return torch.sum(torch.stack(choice_log_probs))

def run_sensitivity_computation():
    task_id = "phase2_sensitivity"
    if is_task_completed(task_id):
        print("[Phase 2] Sensitivity already computed. Skipping.")
        return

    ensure_dirs()
    
    print("[Phase 2] Loading dataset D_screen...")
    ds_path = os.path.join(DATA_DIR, "D_screen.pt")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Missing {ds_path}. Run phase1 first.")
    d_screen = torch.load(ds_path)
    
    print(f"[Phase 2] Loading model {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        revision=HF_REVISION,
        torch_dtype=DTYPE, 
        device_map=DEVICE, 
        trust_remote_code=True
    )
    
    # Freeze all model parameters EXCEPT the target layer
    # This prevents the backward pass from propagating further back, massive speedup!
    for name, param in model.named_parameters():
        if f"layers.{TARGET_LAYER_IDX}." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    
    # Patch the target layer experts individually
    experts_module = model.model.layers[TARGET_LAYER_IDX].mlp.experts
    experts_module._captured_activations = []
    
    for e_idx, expert in enumerate(experts_module):
        expert._expert_idx = e_idx
        expert._global_storage = experts_module._captured_activations
        # Allow mask_config to be passed down if needed in Phase 4
        expert._mask_config = getattr(experts_module, "_mask_config", None)
        expert.forward = types.MethodType(patched_expert_forward, expert)
    
    t_k_path = os.path.join(SENSITIVITY_DIR, "T_K.npy")
    a_n_path = os.path.join(SENSITIVITY_DIR, "A_n.npy")
    w_n_path = os.path.join(SENSITIVITY_DIR, "W_n.npy")
    
    if os.path.exists(t_k_path) and os.path.exists(a_n_path):
        print("[Phase 2] Found intermediate checkpoints for T_K and A_n. Skipping the 1000-example loop.")
        t_k_scores = np.load(t_k_path)
        a_n_scores = np.load(a_n_path)
    else:
        # Prepare accumulators
        # T_K = sum |a * grad(a)|
        # A_n = sum |a|
        t_k_accum = torch.zeros((N_EXPERTS, INTERMEDIATE_SIZE), device=DEVICE, dtype=torch.float32)
        a_n_accum = torch.zeros((N_EXPERTS, INTERMEDIATE_SIZE), device=DEVICE, dtype=torch.float32)
        total_tokens = torch.zeros(N_EXPERTS, device=DEVICE, dtype=torch.float32)
        
        model.train() # Need gradients, though we don't update weights
        
        print(f"[Phase 2] Computing sensitivity over {len(d_screen)} examples...")
        for idx, item in enumerate(d_screen):
            if idx % 10 == 0:
                print(f"  Processing {idx}/{len(d_screen)}")
                
            model.zero_grad()
            experts_module._captured_activations.clear()
            
            prompt, candidates, correct_idx = format_arc_question(item)
            
            # Compute logprobs for each choice
            log_probs = []
            for choice in candidates:
                log_probs.append(compute_choice_logprob(model, tokenizer, prompt, choice))
                
            log_probs = torch.stack(log_probs)
            probs = torch.nn.functional.softmax(log_probs, dim=0)
            
            # Capability surrogate loss L_K = -log P(correct)
            loss = -torch.log(probs[correct_idx] + 1e-8)
            
            loss.backward()
            
            # Extract gradients and accumulate safely without building a new graph
            with torch.no_grad():
                for capture in experts_module._captured_activations:
                    expert_idx = capture["expert_idx"]
                    tensor = capture["tensor"] # (batch_tokens, intermediate_size)
                    
                    if tensor.grad is not None:
                        # Taylor score: |a * grad(a)|
                        taylor = torch.abs(tensor.detach() * tensor.grad).sum(dim=0).float()
                        act_mag = torch.abs(tensor.detach()).sum(dim=0).float()
    
                        
                        t_k_accum[expert_idx] += taylor
                        a_n_accum[expert_idx] += act_mag
                        total_tokens[expert_idx] += capture["num_tokens"]
                        
                    # Explicitly clear gradients to free memory immediately
                    tensor.grad = None
                
            # Clear the list to free memory for the next iteration
            experts_module._captured_activations.clear()
    
        # Normalize by total tokens processed per expert (or globally?)
        # "E[...]" implies expectation. We normalize by the number of tokens routed to that expert.
        safe_tokens = torch.clamp(total_tokens, min=1.0).unsqueeze(1)
        t_k_scores = (t_k_accum / safe_tokens).cpu().numpy()
        a_n_scores = (a_n_accum / safe_tokens).cpu().numpy()
        
        print("[Phase 2] Saving intermediate sensitivity checkpoints...")
        np.save(t_k_path, t_k_scores)
        np.save(a_n_path, a_n_scores)
    
    # ══════════════════════════════════════════════════════════
    # Weight Magnitudes
    # ══════════════════════════════════════════════════════════
    print("[Phase 2] Computing weight magnitudes...")
    # Using L2 norm of the down_proj weights for each neuron.
    # down_proj weight shape is usually (hidden_size, intermediate_size) 
    # For OLMoE it's a batched parameter or individual. Let's inspect shape.
    w_n_scores = np.zeros((N_EXPERTS, INTERMEDIATE_SIZE), dtype=np.float32)
    for e in range(N_EXPERTS):
        # Robustly extract down_proj weight
        wt = None
        try:
            wt = experts_module[e].down_proj.weight.detach().float()
        except AttributeError:
            expert = experts_module[e]
            # Search for down_proj weight in the expert's parameters
            for name, param in expert.named_parameters():
                if 'down_proj' in name:
                    wt = param.detach().float()
                    break
            
            # If not found inside expert, maybe it's a fused module
            if wt is None and hasattr(experts_module, 'down_proj'):
                if isinstance(experts_module.down_proj, torch.nn.ParameterList) or isinstance(experts_module.down_proj, torch.nn.ModuleList):
                    if hasattr(experts_module.down_proj[e], 'weight'):
                        wt = experts_module.down_proj[e].weight.detach().float()
                    else:
                        wt = experts_module.down_proj[e].detach().float()
                elif hasattr(experts_module.down_proj, 'weight'):
                    wt = experts_module.down_proj.weight[e].detach().float()
        
        if wt is None:
            raise RuntimeError(f"Could not locate down_proj weight for expert {e}. Expert type: {type(experts_module[e])}")
            
        # L2 norm across the dimension that maps to the hidden size (typically dim 0)
        norm = torch.norm(wt, p=2, dim=0).cpu().numpy()
        w_n_scores[e] = norm

    # Save results
    np.save(os.path.join(SENSITIVITY_DIR, "T_K.npy"), t_k_scores)
    np.save(os.path.join(SENSITIVITY_DIR, "A_n.npy"), a_n_scores)
    np.save(os.path.join(SENSITIVITY_DIR, "W_n.npy"), w_n_scores)
    
    print("[Phase 2] Sensitivity computation complete.")
    mark_task(task_id, "completed")

if __name__ == "__main__":
    run_sensitivity_computation()
