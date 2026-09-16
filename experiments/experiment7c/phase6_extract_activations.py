"""
EXPERIMENT 7C — PHASE 6: NEURON ACTIVATION EXTRACTION
======================================================
Extracts post-activation neuron signatures for target Layer-8 experts
over a deterministic Wikitext probe sequence.

Uses PyTorch's register_forward_hook on the MoE block to intercept 
hidden_states and routing decisions reliably, regardless of whether
the model uses monkey-patchable expert sub-modules or not.
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import from experiment7b
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.experiment7b.config import (
    TARGET_LAYER_IDX, DEVICE, MAX_SEQ_LEN
)
from experiments.experiment7b.utils.model_utils import (
    load_base_model, get_target_moe_block, cleanup_vram
)
from experiments.experiment7b.utils.evaluation import prepare_wikitext_eval_batches

# 7C Results Directory
RESULTS_DIR_7C = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7c'))

# Number of tokens to extract.
# Defaults to 512 for sanity test, but can be overridden by environment variable for full extraction.
NUM_TOKENS_TO_EXTRACT = int(os.environ.get("CARE_7C_TOKENS", 512))
RANDOM_SEED = 42


def run_extraction():
    print("=" * 70)
    print("EXPERIMENT 7C — PHASE 6: NEURON ACTIVATION EXTRACTION (hook-based)")
    print(f"Targeting: {NUM_TOKENS_TO_EXTRACT} tokens")
    print("=" * 70)

    # 1. Target Identification
    merges_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7b', 'merges', 'actual_merge_results.csv')
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Missing 7B actual merge results: {merges_path}")

    cand_df = pd.read_csv(merges_path)
    target_experts = set(int(e) for e in cand_df["expert_i"].tolist() + cand_df["expert_j"].tolist())
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
    dtype = np.float32

    print(f"[Phase 6] Creating memory-mapped tensor at {memmap_path}")
    print(f"[Phase 6] Expected shape: {shape} (~{np.prod(shape) * 4 / (1024**3):.2f} GB)")

    activations = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)

    with open(memmap_meta_path, "w") as f:
        json.dump({
            "shape": list(shape),
            "dtype": "float32",
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

    # Shared mutable state for the hook
    state = {"global_token_offset": 0}

    # 4. Register a forward hook on the MoE block itself.
    #    This fires BEFORE the block's output is returned.
    #    We intercept hidden_states from the INPUT and routing from 
    #    an inner hook on the experts module.
    #
    #    Strategy: hook on moe_block.experts (OlmoeExperts / similar),
    #    capturing the inputs so we have (hidden_states, top_k_ids, top_k_weights).
    #    Then we manually compute post-activation signatures for target experts only.
    experts_module = moe_block.experts

    def experts_forward_hook(module, inputs, output):
        """
        inputs: tuple of (hidden_states, top_k_ids, top_k_weights)
          - hidden_states: [num_tokens, hidden_dim]  (tokens flattened)
          - top_k_ids: [num_tokens, top_k]           (expert indices per token)
          - top_k_weights: [num_tokens, top_k]

        We compute post-activation signatures for target experts and stream to memmap.
        """
        with torch.no_grad():
            hidden_states = inputs[0]       # [T, H]
            top_k_ids = inputs[1]           # [T, top_k]  int tensor

            T = hidden_states.shape[0]
            offset = state["global_token_offset"]

            for expert_id in target_experts:
                # Find which tokens are routed to this expert
                token_mask = (top_k_ids == expert_id).any(dim=-1)  # [T] bool
                if not token_mask.any():
                    continue

                token_positions = token_mask.nonzero(as_tuple=True)[0]  # [n]
                routed_hidden = hidden_states[token_positions]            # [n, H]

                # Compute post-activation intermediate representation
                # OLMoE uses fused gate_up_proj: [num_experts, 2*ffn_dim, hidden_dim]
                gate_up = torch.nn.functional.linear(routed_hidden, module.gate_up_proj[expert_id])  # [n, 2*ffn_dim]
                gate, up = gate_up.chunk(2, dim=-1)
                post_act = module.act_fn(gate) * up   # [n, ffn_dim=1024]

                # Map local token positions to global positions
                global_positions = offset + token_positions.cpu().numpy()

                # Clip out-of-range positions
                valid = global_positions < NUM_TOKENS_TO_EXTRACT
                if not valid.any():
                    continue

                gp = global_positions[valid]
                vals = post_act[valid].float().cpu().numpy().T   # [1024, n_valid]

                m_idx = expert_idx_to_memmap_idx[expert_id]
                activations[m_idx, :, gp] = vals

    # Register the hook
    hook_handle = experts_module.register_forward_hook(experts_forward_hook)
    model.eval()

    # 5. Execute Streaming Inference
    print(f"\n[Phase 6] Commencing forward passes for {len(eval_chunks)} sequences...")

    with torch.no_grad():
        for chunk in tqdm(eval_chunks, desc="Extracting Signatures"):
            input_ids = chunk["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = chunk["attention_mask"].unsqueeze(0).to(DEVICE)

            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Increment global offset by the number of tokens in this chunk
            state["global_token_offset"] += input_ids.shape[1]

            # Periodically flush memmap
            if state["global_token_offset"] % 50000 < input_ids.shape[1]:
                activations.flush()

            if state["global_token_offset"] >= NUM_TOKENS_TO_EXTRACT:
                break

    # Cleanup
    hook_handle.remove()
    activations.flush()

    # --- Sanity Diagnostics ---
    print("\n[Phase 6] Post-extraction diagnostics:")
    any_nonzero = False
    for ei, mi in expert_idx_to_memmap_idx.items():
        sig = np.array(activations[mi])
        nz = np.count_nonzero(sig)
        mn = sig.min()
        mx = sig.max()
        print(f"  Expert {ei:3d} | nonzero={nz:>10,} | min={mn:.4f} | max={mx:.4f}")
        if nz > 0:
            any_nonzero = True

    if not any_nonzero:
        print("\n[CRITICAL WARNING] ALL activation signatures are zero!")
        print("  Possible causes:")
        print("  1. Hook did not fire (experts_module path is wrong).")
        print("  2. None of the target experts received any tokens.")
        print("  3. gate_up_proj attribute name mismatch.")
        print("  Run the debug script to inspect model architecture.")
    else:
        print("\n[Phase 6] ✓ Non-zero activations confirmed — extraction looks healthy.")

    del activations
    cleanup_vram()

    print("\n[Phase 6] Extraction complete!")
    print(f"Signatures written to {memmap_path}")


if __name__ == "__main__":
    run_extraction()
