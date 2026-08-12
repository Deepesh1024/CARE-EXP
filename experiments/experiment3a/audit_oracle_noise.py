"""
CARE-MoE: Oracle Noise and Sensitivity Audit
==============================================
Runs the Oracle KL calculation on a subset of expert pairs repeatedly,
each time using a different subset of calibration sequences, to measure
the inherent noise/variance in the Oracle metric.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch

# HOTFIX: PyTorch 2.1.1 compatibility for newer Transformers (OLMoE)
if not hasattr(torch, "library"):
    class DummyLibrary:
        pass
    torch.library = DummyLibrary()
if not hasattr(torch.library, "register_fake"):
    def dummy_register_fake(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    torch.library.register_fake = dummy_register_fake

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

# Ensure we can import from exp1
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, "..", "experiment1")))
from CARE_MoE_V3_E1 import find_moe_layers, run_oracle_pair

BASE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
DEVICE = "cpu"
DTYPE = torch.float32

N_PAIRS = 10
N_REPEATS = 5
CALIB_SEQUENCES_PER_RUN = 8
SEQ_LEN = 512

def get_layer_block(model, layer_name, all_layers):
    if layer_name == "first":
        return all_layers[0][1]
    raise ValueError("Only 'first' layer supported for this audit.")

def main():
    print("=" * 70)
    print("ORACLE SENSITIVITY AUDIT")
    print("=" * 70)

    # 1. Load model and tokenizer
    print(f"Loading {BASE_MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=DTYPE
    ).eval()
    model.to(DEVICE)
    
    moe_layers = find_moe_layers(model)
    target_block = get_layer_block(model, "first", moe_layers)

    # 2. Get random expert pairs
    rng = random.Random(42)
    all_pairs = [(i, j) for i in range(64) for j in range(i+1, 64)]
    sample_pairs = rng.sample(all_pairs, N_PAIRS)
    print(f"Selected {N_PAIRS} random pairs for testing.")

    # 3. Get calibration subsets
    print("Generating calibration subsets from wikitext...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["text"] if len(t.strip()) > 0]
    
    subsets = []
    for r in range(N_REPEATS):
        # Pick different texts for each repeat
        sub_texts = rng.sample(texts, CALIB_SEQUENCES_PER_RUN)
        enc = tokenizer(sub_texts, truncation=True, max_length=SEQ_LEN, padding="max_length", return_tensors="pt")
        subsets.append((enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)))

    # 4. Measure KL multiple times
    results = []
    
    for i, j in tqdm(sample_pairs, desc="Expert Pairs"):
        kl_measurements = []
        for r in range(N_REPEATS):
            input_ids, attention_mask = subsets[r]
            oracle_res = run_oracle_pair(
                model=model,
                moe_block=target_block,
                i=i,
                j=j,
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_size=2,
                top_k=8
            )
            kl_measurements.append(oracle_res["Oracle_KL"])
            
        kl_arr = np.array(kl_measurements)
        mean_kl = np.mean(kl_arr)
        std_kl = np.std(kl_arr)
        cov = std_kl / (mean_kl + 1e-10)
        max_diff = np.max(kl_arr) - np.min(kl_arr)
        rel_diff = max_diff / (mean_kl + 1e-10)
        
        results.append({
            "Expert_A": i,
            "Expert_B": j,
            "Measurements": kl_measurements,
            "Mean": mean_kl,
            "Std": std_kl,
            "CoV": cov,
            "Max_Abs_Diff": max_diff,
            "Relative_Diff": rel_diff
        })

    # 5. Summarize and output
    df = pd.DataFrame(results)
    
    report_lines = [
        "# EXPERIMENT 3A: ORACLE SENSITIVITY AUDIT",
        "",
        "## Overview",
        "We measured the true Oracle KL for a subset of 10 random expert pairs from the first layer.",
        f"Each pair was measured {N_REPEATS} times. Each repeat used a uniquely sampled subset",
        f"of {CALIB_SEQUENCES_PER_RUN} calibration sequences ({SEQ_LEN} tokens each) from Wikitext.",
        "",
        "## Summary Statistics",
        f"- **Average Coefficient of Variation (CoV)**: {df['CoV'].mean():.4f}",
        f"- **Max CoV observed**: {df['CoV'].max():.4f}",
        f"- **Average Relative Difference (Max-Min / Mean)**: {df['Relative_Diff'].mean():.4f}",
        f"- **Max Absolute Difference**: {df['Max_Abs_Diff'].max():.6f}",
        "",
        "## Conclusion",
    ]
    
    if df['CoV'].mean() < 0.1:
        report_lines.append("The Oracle KL measurement is robust and highly insensitive to the specific calibration batch. The noise floor is negligible.")
    elif df['CoV'].mean() < 0.3:
        report_lines.append("The Oracle KL measurement shows moderate sensitivity to the calibration batch. While variance exists, the mean signal is stable.")
    else:
        report_lines.append("The Oracle KL measurement is highly sensitive to the calibration batch. A significant noise floor exists.")

    report_lines.append("")
    report_lines.append("## Raw Measurements")
    for _, row in df.iterrows():
        report_lines.append(f"- Pair ({row['Expert_A']}, {row['Expert_B']}): Mean={row['Mean']:.6f}, Std={row['Std']:.6f}, CoV={row['CoV']:.2%}")
        
    out_path = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "results", "exp3a_oracle_sensitivity.md"))
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nAudit complete. Saved to {out_path}")

if __name__ == "__main__":
    main()
