"""
Debug script: inspect OLMoE MoE block architecture and verify hook approach.
Run with: python experiments/experiment7c/debug_architecture.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.experiment7b.config import TARGET_LAYER_IDX, DEVICE
from experiments.experiment7b.utils.model_utils import load_base_model, get_target_moe_block

def main():
    print("Loading model for architecture inspection...")
    model, tokenizer = load_base_model()
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)

    print(f"\n--- MoE Block type: {type(moe_block).__name__} ---")
    print("MoE Block children:")
    for name, module in moe_block.named_children():
        print(f"  [{name}]: {type(module).__name__}")

    # Inspect individual OlmoeMLP (expert 0)
    expert_0 = moe_block.experts[0]
    print(f"\n--- Individual Expert type: {type(expert_0).__name__} ---")
    print("Expert 0 children:")
    for name, module in expert_0.named_children():
        print(f"  [{name}]: {type(module).__name__}", end="")
        if hasattr(module, 'weight'):
            print(f"  weight.shape={module.weight.shape}", end="")
        print()

    print("\nKey attribute check on expert 0:")
    for attr in ['gate_proj', 'up_proj', 'down_proj', 'act_fn']:
        has = hasattr(expert_0, attr)
        val = getattr(expert_0, attr, None)
        print(f"  {attr}: exists={has}  type={type(val).__name__}")

    top_k = getattr(moe_block, 'top_k', getattr(moe_block, 'num_experts_per_tok', 'NOT FOUND'))
    print(f"\ntop_k (routing): {top_k}")
    print(f"router type: {type(moe_block.router).__name__}")

    # Test hook on moe_block
    print("\n--- Hook fire test on moe_block ---")
    fired = [False]

    def test_hook(module, inputs, output):
        fired[0] = True
        flat_h = inputs[0].view(-1, inputs[0].shape[-1])
        print(f"  ✓ moe_block hook fired!")
        print(f"    hidden_states shape (flat): {flat_h.shape}")
        # Re-run router
        with torch.no_grad():
            router_logits = module.router(flat_h)
            top_k_ids = torch.topk(router_logits, k=2, dim=-1).indices
            print(f"    router_logits shape: {router_logits.shape}")
            print(f"    top_k_ids shape: {top_k_ids.shape}")
            print(f"    unique experts activated: {top_k_ids.unique().tolist()}")

        # Test post-act computation for expert 0
        expert = module.experts[0]
        token_mask = (top_k_ids == 0).any(dim=-1)
        if token_mask.any():
            routed = flat_h[token_mask]
            gate = expert.act_fn(expert.gate_proj(routed))
            up = expert.up_proj(routed)
            post_act = gate * up
            print(f"    Expert 0 post_act shape: {post_act.shape}  (should be [n, 1024])")
        else:
            print("    Expert 0 received no tokens in this test batch.")

    handle = moe_block.register_forward_hook(test_hook)
    dummy_ids = tokenizer("Hello world this is a test sentence for hook verification.", return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        model(input_ids=dummy_ids)
    handle.remove()

    if not fired[0]:
        print("  ✗ Hook did NOT fire — inspect model architecture further.")
    else:
        print("\n  ✓ Phase 6 hook approach is valid!")

if __name__ == "__main__":
    main()
