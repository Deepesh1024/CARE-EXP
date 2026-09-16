"""
EXPERIMENT 7C — PHASE 6: NEURON ACTIVATION EXTRACTION
======================================================
Extracts post-activation neuron signatures for target Layer-8 experts
over a deterministic Wikitext probe sequence.

Architecture discovery (OLMoE):
- moe_block: OlmoeSparseMoeBlock
  - moe_block.router: OlmoeTopKRouter (single Linear layer)
  - moe_block.experts: nn.ModuleList of 64 OlmoeMLP objects
    - Each OlmoeMLP: gate_proj, up_proj, down_proj, act_fn

Strategy:
  Register a forward_hook on moe_block itself. In the hook, re-run the
  router (one cheap matrix multiply) to get routing decisions, then
  compute post-activation signatures for target experts from their
  individual gate_proj / up_proj / act_fn.
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.experiment7b.config import TARGET_LAYER_IDX, DEVICE
from experiments.experiment7b.utils.model_utils import (
    load_base_model, get_target_moe_block, cleanup_vram
)
from experiments.experiment7b.utils.evaluation import prepare_wikitext_eval_batches

RESULTS_DIR_7C = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp7c'))
NUM_TOKENS_TO_EXTRACT = int(os.environ.get("CARE_7C_TOKENS", 512))
RANDOM_SEED = 42


def run_extraction():
    print("=" * 70)
    print("EXPERIMENT 7C — PHASE 6: NEURON ACTIVATION EXTRACTION")
    print(f"Targeting: {NUM_TOKENS_TO_EXTRACT} tokens")
    print("=" * 70)

    # 1. Target Identification
    merges_path = os.path.join(os.path.dirname(__file__), '..', '..',
                               'results', 'exp7b', 'merges', 'actual_merge_results.csv')
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Missing 7B actual merge results: {merges_path}")

    cand_df = pd.read_csv(merges_path)
    target_experts = set(int(e) for e in cand_df["expert_i"].tolist() + cand_df["expert_j"].tolist())
    target_experts_list = sorted(list(target_experts))
    num_target_experts = len(target_experts_list)
    print(f"[Phase 6] Target experts ({num_target_experts}): {target_experts_list}")

    expert_idx_to_memmap_idx = {e: i for i, e in enumerate(target_experts_list)}

    # 2. Setup Data Storage
    out_dir = os.path.join(RESULTS_DIR_7C, "activations")
    os.makedirs(out_dir, exist_ok=True)

    memmap_path = os.path.join(out_dir, "expert_signatures.npy")
    memmap_meta_path = os.path.join(out_dir, "expert_signatures_meta.json")

    shape = (num_target_experts, 1024, NUM_TOKENS_TO_EXTRACT)
    dtype = np.float32
    gb = np.prod(shape) * 4 / (1024 ** 3)

    print(f"[Phase 6] Creating memmap at {memmap_path}")
    print(f"[Phase 6] Shape: {shape}  (~{gb:.2f} GB)")

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
    experts_list = moe_block.experts   # nn.ModuleList of OlmoeMLP

    # Determine top-k from the block (try common attribute names)
    top_k = getattr(moe_block, 'top_k',
            getattr(moe_block, 'num_experts_per_tok', 2))
    print(f"[Phase 6] Router top-k = {top_k}")

    state = {"global_token_offset": 0}

    # 4. Register hook on moe_block itself
    def moe_block_hook(module, inputs, output):
        """
        inputs[0]: hidden_states  [batch, seq_len, hidden_dim]
        Strategy:
          1. Flatten tokens.
          2. Re-run the router (cheap: one Linear) to recover routing.
          3. For each target expert, compute post-activation signatures
             using that expert's gate_proj / up_proj / act_fn.
          4. Stream results to memmap indexed by global token position.
        """
        with torch.no_grad():
            hidden_states = inputs[0]   # [B, S, H]
            B, S, H = hidden_states.shape
            flat_h = hidden_states.reshape(-1, H)   # [T, H]
            T = flat_h.shape[0]

            # Re-run router to get routing indices
            router_logits = module.router(flat_h)   # [T, num_experts]
            top_k_indices = torch.topk(router_logits, k=top_k, dim=-1).indices  # [T, top_k]

            offset = state["global_token_offset"]

            for eid in target_experts:
                # Find which tokens (within this batch) are routed to expert eid
                token_mask = (top_k_indices == eid).any(dim=-1)   # [T] bool
                if not token_mask.any():
                    continue

                local_positions = token_mask.nonzero(as_tuple=True)[0]   # [n]
                routed_h = flat_h[local_positions]                         # [n, H]

                # Compute post-activation signature via this expert's MLP
                expert = experts_list[eid]
                gate = expert.act_fn(expert.gate_proj(routed_h))   # [n, ffn_dim]
                up   = expert.up_proj(routed_h)                    # [n, ffn_dim]
                post_act = gate * up                                # [n, ffn_dim]

                # Map to global token positions
                global_positions = offset + local_positions.cpu().numpy()
                valid = global_positions < NUM_TOKENS_TO_EXTRACT
                if not valid.any():
                    continue

                gp   = global_positions[valid]
                vals = post_act[valid].float().cpu().numpy().T      # [1024, n_valid]

                m_idx = expert_idx_to_memmap_idx[eid]
                activations[m_idx, :, gp] = vals

    hook_handle = moe_block.register_forward_hook(moe_block_hook)
    model.eval()

    # 5. Execute Streaming Inference
    print(f"\n[Phase 6] Running forward passes over {len(eval_chunks)} sequences...")

    with torch.no_grad():
        for chunk in tqdm(eval_chunks, desc="Extracting Signatures"):
            input_ids = chunk["input_ids"].unsqueeze(0).to(DEVICE)

            _ = model(input_ids=input_ids)

            state["global_token_offset"] += input_ids.shape[1]

            if state["global_token_offset"] % 50000 < input_ids.shape[1]:
                activations.flush()

            if state["global_token_offset"] >= NUM_TOKENS_TO_EXTRACT:
                break

    hook_handle.remove()
    activations.flush()

    # 6. Post-extraction diagnostics
    print("\n[Phase 6] Diagnostics:")
    any_nonzero = False
    for ei, mi in expert_idx_to_memmap_idx.items():
        sig = np.array(activations[mi])
        nz = np.count_nonzero(sig)
        mn, mx = sig.min(), sig.max()
        print(f"  Expert {ei:3d} | nonzero={nz:>10,} | min={mn:+.4f} | max={mx:+.4f}")
        if nz > 0:
            any_nonzero = True

    if not any_nonzero:
        print("\n[CRITICAL] ALL signatures are zero — hook may not have fired.")
    else:
        print("\n[Phase 6] ✓ Non-zero activations confirmed.")

    del activations
    cleanup_vram()

    print(f"\n[Phase 6] Done. Signatures written to {memmap_path}")


if __name__ == "__main__":
    run_extraction()
