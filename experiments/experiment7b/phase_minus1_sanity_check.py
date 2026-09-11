import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, TARGET_LAYER_IDX
from utils.model_utils import load_base_model, get_target_moe_block, apply_in_place_merge, cleanup_vram
from utils.evaluation import prepare_wikitext_eval_batches, collect_baseline_wikitext_logits, evaluate_wikitext_oracle_kl

def main():
    print("Loading Exp 1 results...")
    with open(os.path.join(os.path.dirname(__file__), "../../results/exp1/output.json"), "r") as f:
        d = json.load(f)
    
    pairs = [r for r in d["results"] if r["Seq_Len"] == 512 and r["Layer"] == "middle" and (int(r["Expert_A"]) == 0 and int(r["Expert_B"]) in [1, 2])]
    if not pairs:
        print("Pairs not found!")
        sys.exit(1)
        
    model, tokenizer = load_base_model()
    moe_block = get_target_moe_block(model, TARGET_LAYER_IDX)
    
    print("Preparing 512 sequences of Wikitext-2...")
    # Using 262144 tokens = 512 sequences of length 512
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, max_tokens=262144, seq_len=512)
    
    print("Collecting baseline logits...")
    baseline_logprobs = collect_baseline_wikitext_logits(model, eval_chunks)
    
    results = []
    for pair in pairs:
        i, j = int(pair["Expert_A"]), int(pair["Expert_B"])
        orig_kl = float(pair["Oracle_KL"])
        print(f"\nTesting pair ({i}, {j}) - Exp 1 Oracle_KL: {orig_kl}")
        
        restore_fn = apply_in_place_merge(moe_block, i, j)
        try:
            actual_kl = evaluate_wikitext_oracle_kl(model, eval_chunks, baseline_logprobs)
            abs_err = abs(actual_kl - orig_kl)
            rel_err = abs_err / max(abs(orig_kl), 1e-12)
            print(f"Computed D_actual_KL: {actual_kl}")
            print(f"Absolute Error: {abs_err}")
            print(f"Relative Error: {rel_err}")
            results.append({"i": i, "j": j, "oracle_kl": orig_kl, "actual_kl": actual_kl, "abs_err": abs_err, "rel_err": rel_err})
        finally:
            restore_fn()
            
    os.makedirs(os.path.join(os.path.dirname(__file__), "../../results/exp7b"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "../../results/exp7b/sanity_check_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
