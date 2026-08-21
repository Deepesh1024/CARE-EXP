from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
for name, module in model.named_modules():
    if "router" in name.lower() or "moe" in name.lower():
        print(name, type(module))
