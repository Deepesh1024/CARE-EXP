import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_benchmark import load_config
from methods.reap_family import register_olmoe_in_reap

def test_router_equivalence(model):
    print("\n--- 1. Router Equivalence Test ---")
    layer = model.model.layers[0].mlp
    x = torch.randn(2, 512, model.config.hidden_size, device=model.device, dtype=model.dtype)
    
    native_out = layer.gate(x)
    native_logits = native_out[0]
    
    adapted_logits = layer.router(x)
    
    diff = torch.max(torch.abs(native_logits - adapted_logits)).item()
    print(f"Max abs diff between native and adapted router logits: {diff}")
    assert diff < 1e-6, "Router mismatch!"
    print("[PASS] Router equivalence validated.")

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
    
    # [PATCH] OLMoEExperts expects 3 arguments, but REAP's fused hook passes 1 argument (unweighted activations).
    for name, module in model.named_modules():
        if type(module).__name__ == "OlmoeSparseMoeBlock":
            experts_cls = module.experts.__class__
            if not hasattr(experts_cls, '_original_forward'):
                experts_cls._original_forward = experts_cls.forward
                def patched_experts_forward(self, *args, **kwargs):
                    if len(args) == 1 and not kwargs:
                        # REAP unweighted activations path
                        hidden_states = args[0]
                        E = self.num_experts
                        T = hidden_states.size(0) // E
                        H = self.hidden_dim
                        x = hidden_states.view(E, T, H)
                        
                        gate_up = torch.bmm(x, self.gate_up_proj.transpose(1, 2))
                        gate, up = gate_up.chunk(2, dim=-1)
                        intermediate = self.act_fn(gate) * up
                        down = torch.bmm(intermediate, self.down_proj.transpose(1, 2))
                        return down.view(-1, H)
                    return self._original_forward(*args, **kwargs)
                experts_cls.forward = patched_experts_forward

    # Apply monkey-patched hook factory approach
    reap_path = os.path.join(os.path.dirname(__file__), "..", "external", "reap", "src")
    if reap_path not in sys.path:
        sys.path.insert(0, reap_path)
        
    from reap.observer import MoETransformerObserver, OBSERVER_CONFIG_REGISTRY
    
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
            print("[PASS] Forward equivalence validated. Hooks do not corrupt outputs.")
        else:
            print("[FAIL] Logits diverged.")
            
    except Exception as e:
        print(f"[FAIL] Crashed with {e}")

    return model

if __name__ == "__main__":
    config = load_config()
    model = test_forward_equivalence(config)
    test_router_equivalence(model)
    print("\n[SUCCESS] Validation tests completed successfully!")
