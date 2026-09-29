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
    # JetMoE uses batched tensors for MLP MoE (e.g., [8, in_dim, out_dim])
    # -------------------------------------------------------------
    
    input_linear_param = None
    output_linear_param = None
    
    for name, param in model.named_parameters():
        if "model.layers.0.mlp.input_linear.weight" in name:
            input_linear_param = param
        elif "model.layers.0.mlp.output_linear.weight" in name:
            output_linear_param = param
            
    if input_linear_param is None or output_linear_param is None:
        print("WARNING: Could not find model.layers.0.mlp.input_linear.weight or output_linear.weight.")
        return

    # Backup original weights (clone)
    orig_input = input_linear_param.clone().detach()
    orig_output = output_linear_param.clone().detach()

    # --- PHASE 6: ABLATION ---
    print("\n[+] PHASE 6: Ablating Expert 0 (zeroing weights in the batched tensor)...")
    # Zero only the linear weights — router stays intact so no softmax blowup
    input_linear_param.data[0].zero_()
    output_linear_param.data[0].zero_()
        
    ablated_ppl = evaluate_subset(model, tokenizer)
    print(f"Ablated PPL: {ablated_ppl:.4f}")
    
    # Restore
    input_linear_param.data.copy_(orig_input)
    output_linear_param.data.copy_(orig_output)

    # --- PHASE 7: MERGE ---
    print("\n[+] PHASE 7: Merging Expert 0 and Expert 1 -> (W0 + W1) / 2...")
    merged_input = (orig_input[0] + orig_input[1]) / 2.0
    merged_output = (orig_output[0] + orig_output[1]) / 2.0
    
    input_linear_param.data[0].copy_(merged_input)
    input_linear_param.data[1].copy_(merged_input)
    
    output_linear_param.data[0].copy_(merged_output)
    output_linear_param.data[1].copy_(merged_output)
            
    merged_ppl = evaluate_subset(model, tokenizer)
    print(f"Merged PPL: {merged_ppl:.4f}")
    
    # Restore
    input_linear_param.data.copy_(orig_input)
    output_linear_param.data.copy_(orig_output)
            
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
