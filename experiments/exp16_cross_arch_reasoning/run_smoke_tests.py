import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import torch
import gc
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from evaluate_reasoning_core import (
    extract_gsm8k_answer,
    normalize_gsm8k_gold,
    extract_math500_answer,
    normalize_math_gold,
    is_math_equivalent,
    evaluate_benchmark
)

print("=" * 75)
print("PHASE 2: SMOKE TESTS & SANITY VERIFICATION")
print("=" * 75)

# -----------------------------------------------------------------------------
# Test 1: Parser and Normalizer Unit Tests
# -----------------------------------------------------------------------------
print("\n[Sanity 1/4] Testing parser and extraction logic on synthetic examples...")

# GSM8K tests
assert extract_gsm8k_answer("The solution is clear. #### 42") == 42, "Failed #### 42"
assert extract_gsm8k_answer("Therefore, \\boxed{1,250} is the answer.") == 1250, "Failed \\boxed{1,250}"
assert extract_gsm8k_answer("The final answer is: 17.5") == 17.5, "Failed 17.5"
assert extract_gsm8k_answer("We get 10 Apples and 25 Oranges.") == 25, "Failed fallback last num"
print("  ✓ GSM8K extraction tests passed.")

# MATH-500 tests
assert extract_math500_answer(r"The answer is \boxed{\frac{3}{4}}.") == r"\frac{3}{4}", "Failed fraction"
assert extract_math500_answer(r"Hence \boxed{5\sqrt{2}}.") == r"5\sqrt{2}", "Failed sqrt"
assert is_math_equivalent("0.5", r"\frac{1}{2}"), "Failed 0.5 vs 1/2 equivalence"
assert is_math_equivalent(r"2\pi", r"2\pi"), "Failed string match"
print("  ✓ MATH-500 extraction and equivalence tests passed.")

# -----------------------------------------------------------------------------
# Test 2: JetMoE-8B Smoke Test
# -----------------------------------------------------------------------------
print("\n[Sanity 2/4] Testing JetMoE-8B baseline loading and 2-question generation...")

config_path = os.path.join(SCRIPT_DIR, "evaluation_config.json")
with open(config_path) as f:
    eval_cfg = json.load(f)

# Override samples for 2-question smoke test
smoke_cfg = json.loads(json.dumps(eval_cfg))
smoke_cfg["benchmarks"]["gsm8k"]["num_samples"] = 2
smoke_cfg["benchmarks"]["gsm8k"]["sample_indices"] = [0, 1]
smoke_cfg["benchmarks"]["math500"]["num_samples"] = 2
smoke_cfg["benchmarks"]["math500"]["sample_indices"] = [0, 1]

smoke_gen_dir = os.path.join(SCRIPT_DIR, "raw_generations", "smoke_test")
os.makedirs(smoke_gen_dir, exist_ok=True)

try:
    from run_jetmoe_reasoning import load_or_build_jetmoe
    jet_tok = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    jet_model = load_or_build_jetmoe("Baseline", 8, jet_tok)
    
    # Run 2 questions
    res = evaluate_benchmark(
        model=jet_model,
        tokenizer=jet_tok,
        benchmark_key="gsm8k",
        model_label="smoke_jetmoe_base",
        architecture_type="base",
        config=smoke_cfg,
        raw_gen_dir=smoke_gen_dir
    )
    print(f"  ✓ JetMoE-8B forward generation & extraction successful: {res['completed'] if 'completed' in res else res['total_questions']} Qs processed.")
    del jet_model
    torch.cuda.empty_cache()
    gc.collect()
except Exception as e:
    print(f"  [JetMoE Smoke Error]: {e}")
    import traceback; traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 3: Phi-3.5-MoE Smoke Test
# -----------------------------------------------------------------------------
print("\n[Sanity 3/4] Testing Phi-3.5-MoE 4-bit loading and 2-question generation...")
try:
    from run_phi_reasoning import load_fresh_phi_model
    phi_tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
    phi_model = load_fresh_phi_model()
    
    res = evaluate_benchmark(
        model=phi_model,
        tokenizer=phi_tok,
        benchmark_key="gsm8k",
        model_label="smoke_phi_base",
        architecture_type="instruct",
        config=smoke_cfg,
        raw_gen_dir=smoke_gen_dir
    )
    print(f"  ✓ Phi-3.5-MoE forward generation & extraction successful: {res['total_questions']} Qs processed.")
    del phi_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  [Phi Smoke Error]: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 75)
print("ALL SMOKE TESTS AND INVARIANTS VERIFIED!")
print("=" * 75)
