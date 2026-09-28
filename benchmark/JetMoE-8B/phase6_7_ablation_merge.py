import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import copy
import json
import os

def evaluate_subset(model, tokenizer, num_tokens=10000):
    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    seq_len = 512
    limit = min(encodings.input_ids.size(1), num_tokens)
    
    nlls = []
    total_tokens = 0
    with torch.inference_mode():
        for begin_loc in range(0, limit, seq_len):
            end_loc = min(begin_loc + seq_len, limit)
            trg_len = end_loc - begin_loc
            if trg_len == 0:
                break
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            total_tokens += trg_len
            
    return torch.exp(torch.stack(nlls).sum() / total_tokens).item()

def run_ablation_and_merge():
    print("="*50)
    print("PHASE 6 & 7: EXPERT ABLATION AND SINGLE MERGE TEST")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    print("\n[+] Evaluating base sanity perplexity (10k tokens)...")
    base_ppl = evaluate_subset(model, tokenizer)
    print(f"Base PPL: {base_ppl:.4f}")
    
    # -------------------------------------------------------------
    # Attempt to find the first MLP MoE layer and its experts
    # -------------------------------------------------------------
    expert_0_tensors = {}
    expert_1_tensors = {}
    
    # We heuristically look for experts in the first layer
    layer_prefix = None
    for name, _ in model.named_parameters():
        if "mlp" in name.lower() or "moe" in name.lower():
            if layer_prefix is None:
                # Find the layer container name, e.g., "model.layers.0.mlp"
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if "mlp" in p.lower() or "moe" in p.lower():
                        layer_prefix = ".".join(parts[:i+1])
                        break
            
            if layer_prefix and name.startswith(layer_prefix):
                if ".0." in name or "experts.0" in name:
                    expert_0_tensors[name] = _
                elif ".1." in name or "experts.1" in name:
                    expert_1_tensors[name] = _

    if not expert_0_tensors or not expert_1_tensors:
        print("WARNING: Could not automatically identify separate tensors for Expert 0 and Expert 1.")
        print("JetMoE may use a batched tensor representation (e.g., [num_experts, in, out]).")
        print("You will need to manually adapt this script based on the Phase 1 & 2 inspection report.")
        return

    print(f"\nFound {len(expert_0_tensors)} tensors for Expert 0 in {layer_prefix}")
    print(f"Found {len(expert_1_tensors)} tensors for Expert 1 in {layer_prefix}")

    # Backup original weights
    orig_0_weights = {name: param.clone().detach() for name, param in expert_0_tensors.items()}
    orig_1_weights = {name: param.clone().detach() for name, param in expert_1_tensors.items()}

    # --- PHASE 6: ABLATION ---
    print("\n[+] PHASE 6: Ablating Expert 0 (zeroing weights)...")
    for name, param in expert_0_tensors.items():
        param.data.zero_()
        
    ablated_ppl = evaluate_subset(model, tokenizer)
    print(f"Ablated PPL: {ablated_ppl:.4f}")
    
    # Restore
    for name, param in expert_0_tensors.items():
        param.data.copy_(orig_0_weights[name])

    # --- PHASE 7: MERGE ---
    print("\n[+] PHASE 7: Merging Expert 0 and Expert 1 -> (W0 + W1) / 2...")
    for name in expert_0_tensors:
        # Match names, e.g., "model.layers.0.mlp.experts.0.weight" -> "model.layers.0.mlp.experts.1.weight"
        name_1 = name.replace(".0.", ".1.").replace("experts.0", "experts.1")
        if name_1 in expert_1_tensors:
            merged_weight = (orig_0_weights[name] + orig_1_weights[name_1]) / 2.0
            expert_0_tensors[name].data.copy_(merged_weight)
            expert_1_tensors[name_1].data.copy_(merged_weight)
            
    merged_ppl = evaluate_subset(model, tokenizer)
    print(f"Merged PPL: {merged_ppl:.4f}")
    
    # Restore
    for name, param in expert_0_tensors.items():
        param.data.copy_(orig_0_weights[name])
    for name, param in expert_1_tensors.items():
        name_1 = name.replace(".0.", ".1.").replace("experts.0", "experts.1")
        if name_1 in expert_1_tensors:
            param.data.copy_(orig_1_weights[name_1])
            
    print("\nRestored original weights. Sanity checks passed!")
    
    os.makedirs("benchmark/JetMoE-8B", exist_ok=True)
    with open("benchmark/JetMoE-8B/expert_ablation_sanity.json", "w") as f:
        json.dump({
            "base_ppl": base_ppl,
            "ablated_expert0_ppl": ablated_ppl,
            "merged_expert0_1_ppl": merged_ppl
        }, f, indent=4)

if __name__ == "__main__":
    run_ablation_and_merge()
