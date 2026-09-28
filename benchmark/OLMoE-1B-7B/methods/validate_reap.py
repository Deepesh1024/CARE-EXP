import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.run_benchmark import load_config
from benchmark.methods.reap_family import register_olmoe_in_reap

def test_forward_equivalence(config):
    print("\n--- 2. Forward Equivalence Test ---")
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    
    print("Loading native model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=config['model']['device'],
    )
    
    x = torch.randint(0, 1000, (1, 32), device=model.device)
    with torch.no_grad():
        out_unpatched = model(x).logits
        
    print("Patching model...")
    register_olmoe_in_reap(model)
    
    # We will test the monkey-patched hook factory approach
    from benchmark.external.reap.src.reap.observer import MoETransformerObserver, OBSERVER_CONFIG_REGISTRY
    
    original_hook_factory = MoETransformerObserver._hook_factory
    
    def patched_hook_factory(self, module, layer_number):
        orig_hook_fn = original_hook_factory(self, module, layer_number)
        
        def wrapped_hook_fn(mod, args, output):
            if not isinstance(output, tuple):
                dummy_scores = torch.empty((mod.num_experts, 0), device=output.device if hasattr(output, 'device') else 'cpu')
                fake_output = (output, dummy_scores)
            else:
                fake_output = output
            return orig_hook_fn(mod, args, fake_output)
            
        return wrapped_hook_fn
        
    MoETransformerObserver._hook_factory = patched_hook_factory
    
    # Init observer just to attach the hooks
    model_cls = type(model).__name__
    observer_config = OBSERVER_CONFIG_REGISTRY[model_cls](distance_measure="cosine")
    observer = MoETransformerObserver(model=model, hook_config=observer_config)
            
    print("Running patched model forward pass...")
    try:
        with torch.no_grad():
            out_patched = model(x).logits
            
        diff = torch.max(torch.abs(out_unpatched - out_patched)).item()
        print(f"Max abs diff between unpatched and patched model logits: {diff}")
        if diff < 1e-6:
            print("PASS: Forward equivalence validated. Hooks do not corrupt outputs.")
        else:
            print("FAIL: Logits diverged.")
            
    except Exception as e:
        print(f"FAIL: Crashed with {e}")

if __name__ == "__main__":
    config = load_config()
    test_forward_equivalence(config)
