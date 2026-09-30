"""
Phase 12: Multi-Capability Evaluation
======================================
Evaluates the saved JetMoE-8B checkpoints on:
  1. WikiText-2 (via isolated subprocess worker -- phase12_wikitext_worker.py)
  2. MMLU (via lm-eval subprocess)
  3. GSM8K (via lm-eval subprocess)
  4. HumanEval (via lm-eval subprocess)

Both WikiText and lm-eval run in separate subprocess so the parent process
never holds any GPU memory, guaranteeing complete VRAM release between runs.

Prerequisites:
    pip install lm-eval
    export HF_ALLOW_CODE_EVAL=1
"""

import os
import sys
import json
import time
import subprocess

CKPT_DIR    = "benchmark_results/JetMoE-8B/checkpoints"
RESULTS_DIR = "benchmark_results/JetMoE-8B/multicapability_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to this file's directory so we can find the worker script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_WORKER   = os.path.join(SCRIPT_DIR, "phase12_wikitext_worker.py")
LMEVAL_WORKER = os.path.join(SCRIPT_DIR, "phase12_lmeval_worker.py")


def run_wikitext_subprocess(model_name, model_path, num_tokens=15000):
    """
    Run WikiText-2 PPL eval in a completely isolated subprocess.
    This guarantees the GPU is 100% free after the call returns.
    """
    print(f"    [WikiText-2] Evaluating {model_name} in subprocess...")

    result = subprocess.run(
        [sys.executable, WIKI_WORKER, model_path, str(num_tokens)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"    [WARNING] WikiText eval subprocess failed for {model_name}:")
        # Print last 20 lines of stderr for debugging
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-20:]:
            print(f"      {line}")
        return None

    # Parse the JSON from the last line of stdout (worker prints JSON at the end)
    stdout_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    for line in reversed(stdout_lines):
        try:
            data = json.loads(line)
            print(f"    [WikiText-2] PPL: {data['ppl']:.4f}  VRAM peak: {data['peak_vram_gb']:.2f} GB")
            return data
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"    [WARNING] Could not parse WikiText result for {model_name}")
    return None


def run_lm_eval(model_name, model_path, tasks="mmlu,gsm8k,humaneval"):
    """
    Run lm-eval in a completely isolated subprocess via the lmeval worker.
    The worker pre-initializes CUDA before lm_eval loads to avoid
    the caching_allocator_warmup crash.
    """
    print(f"    [lm-eval] Running {tasks} for {model_name}...")
    output_dir = os.path.join(RESULTS_DIR, f"{model_name}_lmeval")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    # Ensure GPU is visible to the subprocess
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [sys.executable, LMEVAL_WORKER, model_path, output_dir, tasks]

    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"    [lm-eval] Done for {model_name}.")
    except subprocess.CalledProcessError as e:
        print(f"    [WARNING] lm_eval failed for {model_name}: exit code {e.returncode}")
        # Remove empty output dir so pipeline retries on next run
        if os.path.isdir(output_dir):
            json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
            if not json_files:
                import shutil
                shutil.rmtree(output_dir)
                print(f"    [INFO] Removed empty lm-eval dir -- will retry next run.")


def main():
    print("=" * 60)
    print("PHASE 12: MULTI-CAPABILITY EVALUATION")
    print("  (Full subprocess isolation for zero VRAM leaks)")
    print("=" * 60)

    if not os.path.exists(CKPT_DIR):
        print(f"Error: {CKPT_DIR} not found. Please run phase11_save_checkpoints.py first.")
        return

    if not os.path.exists(WIKI_WORKER):
        print(f"Error: WikiText worker not found at {WIKI_WORKER}")
        print("Please ensure phase12_wikitext_worker.py is in the same directory.")
        return

    models = sorted([d for d in os.listdir(CKPT_DIR) if os.path.isdir(os.path.join(CKPT_DIR, d))])

    if not models:
        print(f"Error: No checkpoints found in {CKPT_DIR}.")
        return

    print(f"Found {len(models)} models to evaluate: {models}\n")

    wiki_results_path = os.path.join(RESULTS_DIR, "results_wikitext2.json")
    if os.path.exists(wiki_results_path):
        with open(wiki_results_path, "r") as f:
            wiki_results = json.load(f)
        print(f"Loaded existing WikiText results ({len(wiki_results)} models cached).\n")
    else:
        wiki_results = {}

    for i, model_name in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Evaluating: {model_name}")
        model_path = os.path.join(CKPT_DIR, model_name)

        # ── 1. WikiText-2 (subprocess) ─────────────────────────────────────────
        if model_name not in wiki_results:
            res = run_wikitext_subprocess(model_name, model_path)
            if res is not None:
                wiki_results[model_name] = res
                with open(wiki_results_path, "w") as f:
                    json.dump(wiki_results, f, indent=4)
        else:
            print(f"    [WikiText-2] Already evaluated ({wiki_results[model_name]['ppl']:.4f}), skipping.")

        # Brief pause between subprocess launches to let CUDA fully settle
        time.sleep(2)

        # ── 2. LM Eval (subprocess) ────────────────────────────────────────────
        lmeval_out = os.path.join(RESULTS_DIR, f"{model_name}_lmeval")
        if not os.path.exists(lmeval_out):
            run_lm_eval(model_name, model_path)
        else:
            print(f"    [lm-eval] Already evaluated, skipping.")

        # Brief pause between subprocess launches
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results saved to: {RESULTS_DIR}")
    wiki_done = len(wiki_results)
    lmeval_done = len([d for d in os.listdir(RESULTS_DIR)
                       if d.endswith("_lmeval") and os.path.isdir(os.path.join(RESULTS_DIR, d))
                       and any(f.endswith(".json") for f in os.listdir(os.path.join(RESULTS_DIR, d)))])
    print(f"  WikiText-2: {wiki_done}/{len(models)} models done")
    print(f"  LM-Eval:    {lmeval_done}/{len(models)} models done")


if __name__ == "__main__":
    main()
