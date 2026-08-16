import os
import json
import time
import torch
import numpy as np
import lm_eval
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lm_eval.tasks

def main():
    os.makedirs("results/exp5", exist_ok=True)
    MODEL_ID = "allenai/OLMoE-1B-7B-0924"
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        DEVICE = "mps"

    print(f"Loading model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map=DEVICE if DEVICE == "cuda:0" else None
    )
    if DEVICE == "mps":
        model = model.to(DEVICE)

    # 1. WikiText-2 Perplexity (Validation)
    def evaluate_ppl():
        print("Evaluating WikiText-2 PPL...")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
        
        max_length = model.config.max_position_embeddings
        stride = 512
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean()).item()
        return ppl

    # 2. Latency / Throughput
    def evaluate_efficiency():
        print("Evaluating Latency & Throughput...")
        dummy_input = torch.randint(0, model.config.vocab_size, (1, 512)).to(DEVICE)
        
        print("Warmup (10 passes)...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
                
        print("Measuring (100 passes)...")
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start_t = time.perf_counter()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif torch.backends.mps.is_available():
                    torch.mps.synchronize()
                end_t = time.perf_counter()
                latencies.append((end_t - start_t) * 1000) # ms
                
        latencies = np.array(latencies)
        throughput = 512 / (latencies / 1000) # tokens per sec
        
        return {
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_std_ms": float(np.std(latencies)),
            "latency_median_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "throughput_mean_tps": float(np.mean(throughput)),
            "throughput_std_tps": float(np.std(throughput))
        }

    # Run evaluations
    ppl_val = evaluate_ppl()
    print(f"WikiText-2 PPL: {ppl_val}")

    efficiency = evaluate_efficiency()

from lm_eval.models.huggingface import HFLM

    print("Evaluating ARC & MMLU via lm_eval...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, device=DEVICE)
    task_manager = lm_eval.tasks.TaskManager()
    
    # Evaluate 3 times with different seeds for stochasticity, though MMLU/ARC are loglikelihood
    # The prompt explicitly requested N=3 or multiple times to capture hardware/driver variance
    mmlu_scores = []
    arc_scores = []
    
    for seed in [42, 101, 2024]:
        print(f"Running eval harness with seed {seed}...")
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=["arc_challenge", "mmlu"],
            num_fewshot=0,
            batch_size="auto",
            task_manager=task_manager,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
        )
        arc_scores.append(results['results']['arc_challenge']['acc,none'])
        mmlu_scores.append(results['results']['mmlu']['acc,none'])

    arc_scores = np.array(arc_scores)
    mmlu_scores = np.array(mmlu_scores)

    noise_floor = {
        "model": MODEL_ID,
        "hardware": DEVICE,
        "quality": {
            "wikitext_2_ppl": {
                "mean": ppl_val,
                "std": 0.0,
                "ci_95": [ppl_val, ppl_val],
                "n": 1
            },
            "arc_challenge_acc": {
                "mean": float(np.mean(arc_scores)),
                "std": float(np.std(arc_scores)),
                "ci_95": [float(np.percentile(arc_scores, 2.5)), float(np.percentile(arc_scores, 97.5))],
                "n": 3
            },
            "mmlu_acc": {
                "mean": float(np.mean(mmlu_scores)),
                "std": float(np.std(mmlu_scores)),
                "ci_95": [float(np.percentile(mmlu_scores, 2.5)), float(np.percentile(mmlu_scores, 97.5))],
                "n": 3
            }
        },
        "efficiency": {
            "latency_ms": {
                "mean": efficiency["latency_mean_ms"],
                "std": efficiency["latency_std_ms"],
                "median": efficiency["latency_median_ms"],
                "p95": efficiency["latency_p95_ms"],
                "n": 100
            },
            "throughput_tps": {
                "mean": efficiency["throughput_mean_tps"],
                "std": efficiency["throughput_std_tps"],
                "n": 100
            }
        }
    }

    with open("results/exp5/evaluation_noise_floor.json", "w") as f:
        json.dump(noise_floor, f, indent=4)
        
    print("Noise floor evaluation complete! Results saved to results/exp5/evaluation_noise_floor.json")

if __name__ == "__main__":
    main()
