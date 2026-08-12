import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment1")))
from CARE_MoE_V3_E1 import InPlaceParameterAveragingMerge, find_moe_layers, run_oracle_pair

def main():
    print("[TEST] Running Oracle KL Equivalence Test...")
    model_id = "allenai/OLMoE-1B-7B-0924"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[TEST] Loading model {model_id} on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,  # CPU safe
        device_map="cpu",
        low_cpu_mem_usage=True
    ).eval()
    
    moe_layers = find_moe_layers(model)
    _, first_layer = moe_layers[0]
    
    print("[TEST] Preparing tiny input...")
    text = "This is a tiny test for the oracle equivalence."
    inputs = tokenizer(text, return_tensors="pt")
    
    print("[TEST] Running existing Exp 1 / 3B Oracle on pair (0, 1)...")
    # run_oracle_pair signature: model, moe_block, i, j, input_ids, attention_mask, batch_size, top_k
    try:
        kl_div, _, _ = run_oracle_pair(
            model=model, 
            moe_block=first_layer, 
            i=0, 
            j=1, 
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            batch_size=1, 
            top_k=8
        )
        print(f"[TEST] SUCCESS! Computed KL Divergence: {kl_div}")
    except Exception as e:
        print(f"[TEST] FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
