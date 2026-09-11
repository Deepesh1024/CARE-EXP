"""
EXPERIMENT 7B - PHASE 6: NEURON ACTIVATION EXTRACTION
=====================================================
Extracts raw post-activation neuron signatures for the candidate experts
over a deterministic fixed probe dataset (Wikitext-2), bypassing RAM
by streaming directly to a numpy memmap.
"""

import os
import sys
import types
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CANDIDATE_DIR, TARGET_LAYER_IDX, DEVICE, DTYPE, 
    EVAL_TOKENS_LIMIT, MAX_SEQ_LEN, RESULTS_DIR
)
from utils.model_utils import (
    load_base_model, get_target_moe_block, cleanup_vram
)
from utils.evaluation import prepare_wikitext_eval_batches

# Maximum tokens to extract (configurable here to prevent massive files if only testing)
NUM_TOKENS_TO_EXTRACT = 512

def run_extraction():
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 6: NEURON ACTIVATION EXTRACTION")
    print("=" * 70)

    # 1. Target Identification
    cand_path = os.path.join(CANDIDATE_DIR, "candidate_pairs.csv")
    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Missing candidates file: {cand_path}")
    
    cand_df = pd.read_csv(cand_path)
    # Collect all unique experts from candidate pairs
    target_experts = set(cand_df["expert_i"].tolist() + cand_df["expert_j"].tolist())
    target_experts_list = sorted(list(target_experts))
    num_target_experts = len(target_experts_list)
    print(f"[Phase 6] Target experts to extract ({num_target_experts}): {target_experts_list}")

    # Create mapping from expert ID to memmap index
    expert_idx_to_memmap_idx = {e: i for i, e in enumerate(target_experts_list)}

    # 2. Setup Data Storage
    out_dir = os.path.join(RESULTS_DIR, "activations")
    os.makedirs(out_dir, exist_ok=True)
    
    memmap_path = os.path.join(out_dir, "expert_signatures.npy")
    memmap_meta_path = os.path.join(out_dir, "expert_signatures_meta.json")
    
    # Shape: [num_target_experts, 1024 (neurons), num_tokens]
    # For 36 experts, 1024 neurons, 262144 tokens, this is ~19 GB in float16
    shape = (num_target_experts, 1024, NUM_TOKENS_TO_EXTRACT)
    dtype = np.float16
    
    print(f"[Phase 6] Creating memory-mapped tensor at {memmap_path}")
    print(f"[Phase 6] Expected shape: {shape} (~{np.prod(shape) * 2 / (1024**3):.2f} GB)")
    
    # w+ overwrites existing
    activations = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
    
    # Save metadata so we can read it back easily
    with open(memmap_meta_path, "w") as f:
        json.dump({
            "shape": shape,
            "dtype": "float16",
            "expert_mapping": expert_idx_to_memmap_idx,
            "num_tokens": NUM_TOKENS_TO_EXTRACT
        }, f, indent=2)

    # 3. Model & Data Initialization
    model, tokenizer = load_base_model()
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, max_tokens=NUM_TOKENS_TO_EXTRACT)
    
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)
    experts_module = moe_block.experts

    # Global tracking offset
    global_token_offset = [0]

    # 4. Monkey-Patching for Streaming Interception
    def patched_experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx_val = expert_idx[0].item()
            if expert_idx_val == self.num_experts: continue
                
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx_val])
            current_state = hidden_states[token_idx]
            
            # Forward pass up to activation
            gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx_val]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up  # This is the post-activation neuron value
            
            # --- INTERCEPTION ---
            # If this is a candidate expert we care about, stream to disk
            if expert_idx_val in target_experts:
                m_idx = expert_idx_to_memmap_idx[expert_idx_val]
                
                # token_idx is the index within the current batch (shape: batch_size * seq_len)
                # We need to map it to the global token offset
                global_indices = global_token_offset[0] + token_idx.cpu().numpy()
                
                # Copy values to memmap
                # current_hidden_states shape: [num_routed_tokens, 1024]
                # Transpose to write [neurons, tokens] into [m_idx, :, global_indices]
                values = current_hidden_states.detach().cpu().to(torch.float16).numpy().T
                activations[m_idx, :, global_indices] = values
            # --------------------

            # Complete forward pass
            out = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx_val])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            
            final_hidden_states = final_hidden_states.index_add(0, token_idx, out.to(final_hidden_states.dtype))

        return final_hidden_states

    # Apply patch
    experts_module.forward = types.MethodType(patched_experts_forward, experts_module)
    model.eval()

    # 5. Execute Streaming Inference
    print(f"\n[Phase 6] Commencing forward passes for {len(eval_chunks)} sequences...")
    
    # We pass chunks individually (or in small batches) to control memory and ensure ordering
    with torch.no_grad():
        for chunk in tqdm(eval_chunks, desc="Extracting Signatures"):
            input_ids = chunk["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = chunk["attention_mask"].unsqueeze(0).to(DEVICE)
            
            # Forward pass will trigger the patched hook and stream to disk
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Increment global tracker by the sequence length (assuming fixed length chunks)
            seq_len = input_ids.shape[1]
            global_token_offset[0] += seq_len
            
            # Periodically flush the memmap to disk to be safe
            if global_token_offset[0] % 50000 < seq_len:
                activations.flush()

    # Final flush and cleanup
    activations.flush()
    del activations
    cleanup_vram()
    
    print("\n[Phase 6] Extraction successfully completed!")
    print(f"Signatures securely written to {memmap_path}")

if __name__ == "__main__":
    run_extraction()
