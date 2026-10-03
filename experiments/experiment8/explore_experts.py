import torch
from transformers import AutoModelForCausalLM
import inspect

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924", 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

mlp = model.model.layers[8].mlp
experts = mlp.experts

print(f"Type of experts: {type(experts)}")
print(f"Attributes of experts: {dir(experts)}")

# Print shapes of any parameters inside experts
for name, param in experts.named_parameters():
    print(f"  {name}: {param.shape}")

print("\nForward signature:")
print(inspect.signature(experts.forward))
