import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import types
from tqdm import tqdm

# Add parent directory to path to import from experiment7b
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.experiment7b.config import (
    CANDIDATE_DIR, TARGET_LAYER_IDX, DEVICE,
    EVAL_TOKENS_LIMIT, MAX_SEQ_LEN
)
from experiments.experiment7b.utils.model_utils import (
    load_base_model, get_target_moe_block, cleanup_vram
)
from experiments.experiment7b.utils.evaluation import prepare_wikitext_eval_batches

# 7C Results Directory
RESULTS_DIR_7C = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7c'))

# Number of tokens to extract. 
# SET TO 512 FOR SANITY TEST. 
# SET TO 262144 FOR FULL GPU EXTRACTION.
NUM_TOKENS_TO_EXTRACT = 512
RANDOM_SEED = 42

def run_extraction():
    print("=" * 70)
    print("EXPERIMENT 7C — PHASE 6: NEURON ACTIVATION EXTRACTION")
    print(f"Targeting: {NUM_TOKENS_TO_EXTRACT} tokens")
    print("=" * 70)

    # 1. Target Identification
    # We must extract experts from the 18 candidate pairs of 7B.
    # The actual merge results from 7B contains the exact 18 pairs used.
    merges_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7b', 'merges', 'actual_merge_results.csv')
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Missing 7B actual merge results: {merges_path}")
    
    cand_df = pd.read_csv(merges_path)
    target_experts = set(cand_df["expert_i"].tolist() + cand_df["expert_j"].tolist())
    target_experts_list = sorted(list(target_experts))
    num_target_experts = len(target_experts_list)
    print(f"[Phase 6] Target experts to extract ({num_target_experts}): {target_experts_list}")

    expert_idx_to_memmap_idx = {e: i for i, e in enumerate(target_experts_list)}

    # 2. Setup Data Storage
    out_dir = os.path.join(RESULTS_DIR_7C, "activations")
    os.makedirs(out_dir, exist_ok=True)
    
    memmap_path = os.path.join(out_dir, "expert_signatures.npy")
    memmap_meta_path = os.path.join(out_dir, "expert_signatures_meta.json")
    
    shape = (num_target_experts, 1024, NUM_TOKENS_TO_EXTRACT)
    dtype = np.float16
    
    print(f"[Phase 6] Creating memory-mapped tensor at {memmap_path}")
    print(f"[Phase 6] Expected shape: {shape} (~{np.prod(shape) * 2 / (1024**3):.2f} GB)")
    
    activations = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
    
    with open(memmap_meta_path, "w") as f:
        json.dump({
            "shape": shape,
            "dtype": "float16",
            "expert_mapping": expert_idx_to_memmap_idx,
            "num_tokens": NUM_TOKENS_TO_EXTRACT,
            "seed": RANDOM_SEED,
            "layer_idx": TARGET_LAYER_IDX
        }, f, indent=2)

    # 3. Model & Data Initialization
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    model, tokenizer = load_base_model()
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, max_tokens=NUM_TOKENS_TO_EXTRACT)
    
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)
    experts_module = moe_block.experts

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
            current_hidden_states = self.act_fn(gate) * up  # Post-activation neuron value
            
            # --- INTERCEPTION ---
            if expert_idx_val in target_experts:
                m_idx = expert_idx_to_memmap_idx[expert_idx_val]
                global_indices = global_token_offset[0] + token_idx.cpu().numpy()
                
                # We need to ensure we don't write out of bounds if chunks exceed NUM_TOKENS_TO_EXTRACT
                valid_mask = global_indices < NUM_TOKENS_TO_EXTRACT
                if valid_mask.any():
                    valid_global_indices = global_indices[valid_mask]
                    values = current_hidden_states[valid_mask].detach().cpu().to(torch.float16).numpy().T
                    activations[m_idx, :, valid_global_indices] = values
            # --------------------

            out = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx_val])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states = final_hidden_states.index_add(0, token_idx, out.to(final_hidden_states.dtype))

        return final_hidden_states

    experts_module.forward = types.MethodType(patched_experts_forward, experts_module)
    model.eval()

    # 5. Execute Streaming Inference
    print(f"\n[Phase 6] Commencing forward passes for {len(eval_chunks)} sequences...")
    
    with torch.no_grad():
        for chunk in tqdm(eval_chunks, desc="Extracting Signatures"):
            input_ids = chunk["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = chunk["attention_mask"].unsqueeze(0).to(DEVICE)
            
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            seq_len = input_ids.shape[1]
            global_token_offset[0] += seq_len
            
            if global_token_offset[0] >= NUM_TOKENS_TO_EXTRACT:
                break

    activations.flush()
    del activations
    cleanup_vram()
    
    print("\n[Phase 6] Extraction successfully completed!")
    print(f"Signatures securely written to {memmap_path}")

if __name__ == "__main__":
    run_extraction()
