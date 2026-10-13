import time
import torch
from datasets import load_dataset
from typing import Dict, Any

@torch.no_grad()
def evaluate_wikitext_ppl(
    model,
    tokenizer,
    num_tokens: int = 50000,
    seq_len: int = 512,
    stride: int = 512,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Evaluates WikiText-2 perplexity matching the repository baseline configuration.
    """
    model.eval()
    print(f"[Eval] Loading WikiText-2 test set (evaluating up to {num_tokens} tokens)...", flush=True)
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    total_tokens_available = encodings.input_ids.size(1)
    limit = min(total_tokens_available, num_tokens)
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    nlls = []
    total_tokens = 0
    
    with torch.inference_mode():
        for begin_loc in range(0, limit, stride):
            end_loc = min(begin_loc + seq_len, limit)
            trg_len = end_loc - begin_loc
            if trg_len <= 0:
                break
                
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.item()
            nlls.append(neg_log_likelihood * trg_len)
            total_tokens += trg_len
            del input_ids, target_ids, outputs

    elapsed_time = time.time() - start_time
    import math
    ppl = math.exp(sum(nlls) / total_tokens) if total_tokens > 0 else float("nan")
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
    
    return {
        "perplexity": float(ppl),
        "total_eval_time_sec": float(elapsed_time),
        "peak_vram_gb": float(peak_vram_gb),
        "tokens_per_sec": float(tokens_per_sec),
        "tokens_evaluated": int(total_tokens)
    }
