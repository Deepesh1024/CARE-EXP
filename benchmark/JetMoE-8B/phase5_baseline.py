import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import time
import json
import os
from tqdm import tqdm

def evaluate_baseline():
    print("="*50)
    print("PHASE 5: BASELINE PERPLEXITY")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    seq_len = 512
    max_length = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 2048
    stride = 512
    seq_len = min(seq_len, max_length)

    nlls = []
    total_tokens = 0
    start_time = time.time()
    
    print(f"\nEvaluating Perplexity on Wikitext-2 (stride={stride}, seq_len={seq_len})...")
    
    # Evaluate on a subset to keep execution time reasonable for the benchmark verification
    num_eval_tokens = encodings.input_ids.size(1)
    max_eval_tokens = 50000 # Configurable
    limit = min(num_eval_tokens, max_eval_tokens)
    
    with torch.inference_mode():
        for begin_loc in tqdm(range(0, limit, stride)):
            end_loc = min(begin_loc + seq_len, num_eval_tokens)
            trg_len = end_loc - begin_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood * trg_len)
            total_tokens += trg_len

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"\nBaseline Perplexity: {ppl:.4f}")
    print(f"Total Evaluation Time: {total_time:.2f} s")
    print(f"Peak GPU Memory: {peak_mem:.2f} GB")
    print(f"Tokens / Sec: {total_tokens / total_time:.2f}")

    os.makedirs("benchmark/JetMoE-8B", exist_ok=True)
    with open("benchmark/JetMoE-8B/baseline_metrics.json", "w") as f:
        json.dump({
            "baseline_ppl": ppl,
            "total_eval_time_sec": total_time,
            "peak_gpu_memory_gb": peak_mem,
            "tokens_per_sec": total_tokens / total_time,
            "tokens_evaluated": total_tokens
        }, f, indent=4)

if __name__ == "__main__":
    evaluate_baseline()
