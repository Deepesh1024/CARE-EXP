"""
Phase 12 WikiText Worker
========================
Called as a subprocess by phase12_evaluate_multicapability.py.
Runs WikiText-2 PPL evaluation for a single model and prints JSON result to stdout.
This subprocess isolation guarantees full CUDA memory release between evaluations.

Usage (called automatically by phase12):
    python phase12_wikitext_worker.py <model_path> <num_tokens>
"""

import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    model_path = sys.argv[1]
    num_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 15000

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
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
            target_ids = input_ids.detach().clone()
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            total_tokens += trg_len

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    end_time = time.time()
    eval_time = end_time - start_time
    tokens_per_sec = total_tokens / eval_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Print result as JSON to stdout for the parent process to parse
    print(json.dumps({
        "ppl": ppl,
        "eval_time_sec": eval_time,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb
    }))

if __name__ == "__main__":
    main()
