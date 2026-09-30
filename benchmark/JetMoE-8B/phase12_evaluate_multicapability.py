"""
Phase 12: Multi-Capability Evaluation

Evaluates the saved JetMoE-8B checkpoints on:
1. WikiText-2 (custom protocol to match Phase 8 exactly)
2. MMLU (via lm-eval)
3. GSM8K (via lm-eval)
4. HumanEval (via lm-eval)

Prerequisites:
    pip install lm-eval
    export HF_ALLOW_CODE_EVAL=1
"""

import os
import json
import time
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

CKPT_DIR = "benchmark_results/JetMoE-8B/checkpoints"
RESULTS_DIR = "benchmark_results/JetMoE-8B/multicapability_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_wikitext(model_path, num_tokens=15000):
    print(f"    Evaluating WikiText-2 on {model_path}...")
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
            if trg_len == 0: break
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.detach().clone()
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            total_tokens += trg_len
            
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    
    end_time = time.time()
    eval_time = end_time - start_time
    tokens_per_sec = total_tokens / eval_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {
        "ppl": ppl,
        "eval_time_sec": eval_time,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb
    }

def run_lm_eval(model_name, model_path):
    print(f"    Running lm-eval for MMLU, GSM8K, HumanEval on {model_name}...")
    output_dir = os.path.join(RESULTS_DIR, f"{model_name}_lmeval")
    
    # We use subprocess to ensure clean memory isolation between runs
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
        "--tasks", "mmlu,gsm8k,humaneval",
        "--device", "cuda:0",
        "--batch_size", "1",
        "--output_path", output_dir,
        "--trust_remote_code",
        "--confirm_run_unsafe_code"
    ]
    
    # Important: HumanEval requires HF_ALLOW_CODE_EVAL=1
    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"    [WARNING] lm_eval failed for {model_name}. Ensure lm-eval is installed (pip install lm-eval).")
        print(f"    Error: {e}")

def main():
    print("=" * 60)
    print("PHASE 12: MULTI-CAPABILITY EVALUATION")
    print("=" * 60)
    
    if not os.path.exists(CKPT_DIR):
        print(f"Error: {CKPT_DIR} not found. Please run phase11_save_checkpoints.py first.")
        return
        
    models = sorted([d for d in os.listdir(CKPT_DIR) if os.path.isdir(os.path.join(CKPT_DIR, d))])
    
    if not models:
        print(f"Error: No checkpoints found in {CKPT_DIR}.")
        return
        
    print(f"Found {len(models)} models to evaluate.")
    
    wiki_results_path = os.path.join(RESULTS_DIR, "results_wikitext2.json")
    if os.path.exists(wiki_results_path):
        with open(wiki_results_path, "r") as f:
            wiki_results = json.load(f)
    else:
        wiki_results = {}
        
    for model_name in models:
        print(f"\nEvaluating: {model_name}")
        model_path = os.path.join(CKPT_DIR, model_name)
        
        # 1. WikiText-2 Custom Protocol
        if model_name not in wiki_results:
            try:
                res = evaluate_wikitext(model_path)
                wiki_results[model_name] = res
                # Save incrementally
                with open(wiki_results_path, "w") as f:
                    json.dump(wiki_results, f, indent=4)
                print(f"    WikiText PPL: {res['ppl']:.4f}")
            except Exception as e:
                print(f"    [WARNING] WikiText eval failed for {model_name}: {e}")
        else:
            print("    WikiText-2 already evaluated, skipping.")
            
        # 2. LM Eval (MMLU, GSM8K, HumanEval)
        lmeval_out = os.path.join(RESULTS_DIR, f"{model_name}_lmeval")
        if not os.path.exists(lmeval_out):
            run_lm_eval(model_name, model_path)
        else:
            print("    LM-Eval already run, skipping.")
            
    print("\nEvaluation complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()
