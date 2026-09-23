from transformers import AutoModelForCausalLM
import torch

def inspect():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained('allenai/OLMoE-1B-7B-0924', trust_remote_code=True, device_map='cpu')
    moe = model.model.layers[0].mlp
    experts = moe.experts
    print("\n--- Experts Information ---")
    print("Type:", type(experts))
    print("Has __len__?", hasattr(experts, '__len__'))
    print("Has __getitem__?", hasattr(experts, '__getitem__'))
    
    print("\nDir of experts:")
    print([d for d in dir(experts) if not d.startswith('_')])
    
    if hasattr(experts, 'up_proj'):
        print("up_proj weight shape:", experts.up_proj.weight.shape)
    if hasattr(experts, 'mlp_experts'):
        print("mlp_experts present!")

inspect()
