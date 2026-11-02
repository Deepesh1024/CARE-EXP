"""
benchmark/smoke_test_capabilities.py
====================================
Rapid preflight smoke test verifying the capability harness, probe construction,
and scoring logic on 2 samples per benchmark.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.probes import (
    build_probe_datasets,
    verify_strict_data_separation,
    extract_single_capability_representation,
    build_multi_capability_representation,
    compute_pairwise_distances,
    nominate_candidate_pool,
)
from benchmark.evaluate_capabilities import (
    evaluate_wikitext2_ppl_and_kl,
    evaluate_gsm8k,
    evaluate_mmlu,
    evaluate_mbpp,
)

def run_smoke_test():
    print("[SMOKE TEST] Initializing...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "allenai/OLMoE-1B-7B-0924"

    print(f"[SMOKE TEST] Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[SMOKE TEST] Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()

    print("[SMOKE TEST] Testing WikiText-2 (2 chunks)...")
    ppl, _, logits = evaluate_wikitext2_ppl_and_kl(model, tokenizer, num_eval_chunks=2, device=device)
    print(f"  WikiText-2 PPL: {ppl:.4f}")
    assert ppl > 1.0 and ppl < 100.0, f"Unreasonable PPL: {ppl}"

    print("[SMOKE TEST] Testing GSM8K (2 questions)...")
    gsm_res = evaluate_gsm8k(model, tokenizer, num_samples=2, max_new_tokens=512, device=device)
    print(f"  GSM8K accuracy: {gsm_res['accuracy']}% | Truncations: {gsm_res['truncation_count']}")

    print("[SMOKE TEST] Testing MMLU (4 questions)...")
    mmlu_res = evaluate_mmlu(model, tokenizer, num_samples=4, device=device)
    print(f"  MMLU accuracy: {mmlu_res['accuracy']}%")

    print("[SMOKE TEST] Testing MBPP (2 problems)...")
    mbpp_res = evaluate_mbpp(model, tokenizer, num_samples=2, max_new_tokens=256, device=device)
    print(f"  MBPP pass@1: {mbpp_res['pass_at_1']}%")

    print("[SMOKE TEST] Testing Probe Construction...")
    probes = build_probe_datasets(tokenizer, max_length=256, seed=42)
    print(f"  Probes constructed: {list(probes.keys())}")
    for k, v in probes.items():
        print(f"    - {k}: {len(v)} samples")

    print("\n[SMOKE TEST PASS] All capability components validated successfully!")

if __name__ == "__main__":
    run_smoke_test()
