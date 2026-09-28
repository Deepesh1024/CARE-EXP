import torch
from transformers import AutoModelForCausalLM
import json

def inspect_model():
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA NOT AVAILABLE!")
    
    print("\nLoading jetmoe/jetmoe-8b in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    report = []
    print("\nModule Tree Inspection:")
    
    mlp_moe_location = None
    moa_location = None
    
    for name, module in model.named_modules():
        mod_type = type(module).__name__
        
        is_moe = False
        is_moa = False
        num_experts = None
        top_k = None
        
        # Heuristics to detect MoE/MoA in JetMoE
        if "moe" in name.lower() or "mlp" in name.lower() and hasattr(module, "experts"):
            is_moe = True
            mlp_moe_location = name
            if hasattr(module, "num_experts"):
                num_experts = module.num_experts
            elif hasattr(module, "experts"):
                num_experts = len(module.experts)
            if hasattr(module, "top_k"):
                top_k = module.top_k
                
        if "moa" in name.lower() or "attention" in name.lower() and hasattr(module, "experts"):
            is_moa = True
            moa_location = name
            if hasattr(module, "num_experts"):
                num_experts = module.num_experts
            elif hasattr(module, "experts"):
                num_experts = len(module.experts)
            if hasattr(module, "top_k"):
                top_k = module.top_k
        
        report.append({
            "module_name": name,
            "module_class": mod_type,
            "is_moe": is_moe,
            "is_moa": is_moa,
            "num_experts": num_experts,
            "top_k": top_k
        })
        
    with open("benchmark/JetMoE-8B/jetmoe_architecture.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nExample MLP MoE located at: {mlp_moe_location}")
    print(f"Example MoA located at: {moa_location}")
    
    print("\nFinding EXACT MLP Expert Tensors...")
    for name, param in model.named_parameters():
        if "mlp" in name.lower() and "expert" in name.lower():
            print(f"Found Expert Tensor: {name}, Shape: {param.shape}")
            break # Just print the first one to see the structure

if __name__ == "__main__":
    inspect_model()
