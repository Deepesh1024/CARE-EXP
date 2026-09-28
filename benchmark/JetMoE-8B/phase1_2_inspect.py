import torch
from transformers import AutoModelForCausalLM
import json
import os

def inspect_model():
    print("="*50)
    print("PHASE 0 & 1 & 2: ENVIRONMENT AND MODEL INSPECTION")
    print("="*50)
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA NOT AVAILABLE!")
    
    print("\nLoading jetmoe/jetmoe-8b in BF16 (no gradients)...")
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(
            "jetmoe/jetmoe-8b", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
    
    report = []
    mlp_moe_layers = []
    moa_layers = []
    
    print("\n[+] Inspecting Module Tree...")
    for name, module in model.named_modules():
        mod_type = type(module).__name__
        
        is_moe = "moe" in name.lower() or "mlp" in name.lower()
        is_moa = "moa" in name.lower() or "attention" in name.lower()
        
        # Heuristic check for experts
        has_experts = hasattr(module, "experts") or hasattr(module, "num_experts") or "moe" in mod_type.lower()
        
        num_experts = getattr(module, "num_experts", None)
        if num_experts is None and hasattr(module, "experts"):
            num_experts = len(module.experts) if isinstance(module.experts, (list, torch.nn.ModuleList)) else None
            
        top_k = getattr(module, "top_k", getattr(module, "k", None))
        
        if has_experts:
            if is_moe and not is_moa:
                mlp_moe_layers.append(name)
            elif is_moa:
                moa_layers.append(name)
                
            report.append({
                "module_name": name,
                "module_class": mod_type,
                "is_mlp_moe": is_moe and not is_moa,
                "is_moa": is_moa,
                "num_experts": num_experts,
                "top_k": top_k
            })

    print(f"\nFound {len(mlp_moe_layers)} potential MLP MoE layers.")
    print(f"Found {len(moa_layers)} potential MoA layers.")
    
    # Save the architecture report
    os.makedirs("benchmark/JetMoE-8B", exist_ok=True)
    with open("benchmark/JetMoE-8B/jetmoe_architecture.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n[+] Identifying EXACT MLP Expert Tensors...")
    print("Searching for expert and router parameters...")
    
    tensor_report = {}
    
    # We will pick the first layer block to analyze its structure deeply
    for name, param in model.named_parameters():
        if "layers.0." in name or "block.0." in name or "blocks.0." in name:
            tensor_report[name] = str(list(param.shape))
            if "mlp" in name.lower() or "moe" in name.lower():
                print(f"  {name}: {list(param.shape)} (dtype: {param.dtype})")
                
    with open("benchmark/JetMoE-8B/jetmoe_tensors_layer0.json", "w") as f:
        json.dump(tensor_report, f, indent=4)
        
    print("\nInspection complete! Check 'jetmoe_architecture.json' and 'jetmoe_tensors_layer0.json'.")

if __name__ == "__main__":
    inspect_model()
