"""
Debug script: inspect OLMoE MoE block architecture
to confirm attribute names before running phase6.

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

    print(f"\n--- MoE Block type: {type(moe_block)} ---")
    print("MoE Block children:")
    for name, module in moe_block.named_children():
        print(f"  [{name}]: {type(module).__name__}")

    print(f"\n--- Experts module type: {type(moe_block.experts)} ---")
    print("Experts module children:")
    for name, module in moe_block.experts.named_children():
        print(f"  [{name}]: {type(module).__name__}")

    print("\nExperts module attributes (non-private):")
    for attr in dir(moe_block.experts):
        if not attr.startswith('_'):
            try:
                val = getattr(moe_block.experts, attr)
                if isinstance(val, torch.Tensor):
                    print(f"  {attr}: Tensor {val.shape}")
                elif isinstance(val, torch.nn.Parameter):
                    print(f"  {attr}: Parameter {val.shape}")
            except Exception:
                pass

    # Verify gate_up_proj exists
    print("\n--- Key attribute check ---")
    for attr in ['gate_up_proj', 'gate_proj', 'up_proj', 'down_proj', 'act_fn', 'num_experts']:
        has = hasattr(moe_block.experts, attr)
        val = getattr(moe_block.experts, attr, None)
        shape = val.shape if isinstance(val, (torch.Tensor, torch.nn.Parameter)) else type(val).__name__
        print(f"  {attr}: exists={has}  shape/type={shape}")

    # Test that the hook fires
    print("\n--- Hook fire test ---")
    fired = [False]

    def test_hook(module, inputs, output):
        fired[0] = True
        print(f"  ✓ Hook fired! inputs[0].shape={inputs[0].shape}, inputs[1].shape={inputs[1].shape}")

    handle = moe_block.experts.register_forward_hook(test_hook)

    dummy_ids = tokenizer("Hello world this is a test sentence for hook verification.", return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        model(input_ids=dummy_ids)

    handle.remove()

    if not fired[0]:
        print("  ✗ Hook did NOT fire! The experts sub-module is not called through its forward method.")
        print("  Try registering the hook on moe_block directly or inspect moe_block.forward().")
    else:
        print("  Phase 6 hook approach is valid and will work correctly.")

if __name__ == "__main__":
    main()
