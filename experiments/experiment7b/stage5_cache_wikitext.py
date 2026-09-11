import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# Adjust sys path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.evaluation import prepare_wikitext_eval_batches
from utils.model_utils import load_base_model

def run_stage5():
    print("Running Stage 5: Prepare Wikitext cache...")
    model, tokenizer = load_base_model()
    
    print("Preparing 512 sequences of Wikitext-2...")
    eval_chunks = prepare_wikitext_eval_batches(tokenizer, max_tokens=262144, seq_len=512)
    
    out_dir = os.path.join(os.path.dirname(__file__), "../../results/exp7b/merges")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wikitext_cache.pt")
    
    # Save a simpler dictionary representation to avoid serialization issues
    cache_data = [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in eval_chunks]
    torch.save(cache_data, out_path)
    
    print(f"Stage 5 complete. Saved {len(eval_chunks)} chunks to {out_path}")

if __name__ == "__main__":
    run_stage5()
