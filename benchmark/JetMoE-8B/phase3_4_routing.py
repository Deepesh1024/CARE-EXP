import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

def capture_routing_stats():
    print("="*50)
    print("PHASE 3 & 4: ROUTING HOOK & CALIBRATION DATA")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load fixed calibration dataset (using wikitext to match standard CARE pipeline)
    print("\nLoading calibration dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    texts = [text for text in dataset["text"] if len(text.strip()) > 100]
    calibration_texts = texts[:64] # Start small for RTX 4090
    seq_length = 512
    
    stats = defaultdict(lambda: defaultdict(int))
    
    # -------------------------------------------------------------
    # IMPORTANT: The exact hook depends on JetMoE's router structure.
    # We heuristically hook into the forward pass of modules that 
    # look like MLP MoE routers.
    # -------------------------------------------------------------
    
    def get_router_hook(layer_name):
        def hook(module, input, output):
            # output could be (logits,) or just logits depending on implementation
            # We assume output contains router logits of shape [batch, seq_len, num_experts]
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            if hasattr(logits, "shape") and len(logits.shape) == 3:
                # Calculate routing probabilities and top-k selections
                probs = torch.softmax(logits, dim=-1)
                
                # Assume top_k = 2 based on JetMoE specs
                top_k_probs, top_k_indices = torch.topk(probs, k=2, dim=-1)
                
                # Count expert usage
                flat_indices = top_k_indices.view(-1).tolist()
                for expert_id in flat_indices:
                    stats[layer_name][f"expert_{expert_id}_count"] += 1
        return hook

    hooks = []
    print("\nAttaching hooks to MLP MoE routers...")
    # Find MLP routers (heuristically looking for gate/router in MLP modules)
    for name, module in model.named_modules():
        if "mlp" in name.lower() and ("gate" in name.lower() or "router" in name.lower()):
            if isinstance(module, torch.nn.Linear):
                print(f"Hooking {name}")
                hooks.append(module.register_forward_hook(get_router_hook(name)))

    if not hooks:
        print("WARNING: Could not automatically identify router modules! You may need to manually update the hook logic.")

    print("\nRunning calibration forward passes...")
    with torch.no_grad():
        for text in tqdm(calibration_texts):
            inputs = tokenizer(text, return_tensors="pt", max_length=seq_length, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model(**inputs)

    # Clean up hooks
    for h in hooks:
        h.remove()

    print("\nSaving routing statistics...")
    os.makedirs("benchmark/JetMoE-8B", exist_ok=True)
    with open("benchmark/JetMoE-8B/jetmoe_routing_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Done! Statistics saved to 'jetmoe_routing_stats.json'.")

if __name__ == "__main__":
    capture_routing_stats()
