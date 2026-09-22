"""
CARE-COM v2.2 PPL Recomputation — recomputes PPL at checkpoint experts
for all existing trajectory files without re-running the full compression.

Uses subprocess isolation to avoid VRAM fragmentation across models.

Usage:
    PYTHONPATH=. python -m experiments.care_com_v22.recompute_ppl
"""
import os
import sys
import json
import argparse
import subprocess
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment7b")))
from utils.evaluation import prepare_wikitext_eval_batches

from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.baselines import compute_ppl


def replay_merges_and_compute_ppl(trace, method_name, config, tokenizer, eval_chunks):
    """
    Replays the exact merge sequence from a trace, computing PPL at checkpoints.
    """
    print(f"\n{'='*60}")
    print(f"  Recomputing PPL for: {method_name}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=config.device
    )
    model.eval()
    engine = PhysicalMergeEngine(model)

    ppl_log = {}

    # PPL at 64 (before any merges)
    if engine.current_num_experts in config.checkpoints:
        ppl_log[engine.current_num_experts] = compute_ppl(model, eval_chunks, config)
        print(f"  PPL @ {engine.current_num_experts}: {ppl_log[engine.current_num_experts]:.4f}")

    # Replay each merge step
    for step in trace:
        pair = step["selected_pair"]
        i, j = pair[0], pair[1]

        engine.merge_experts(i, j)

        experts_after = engine.current_num_experts
        if experts_after in config.checkpoints:
            ppl_log[experts_after] = compute_ppl(model, eval_chunks, config)
            print(f"  PPL @ {experts_after}: {ppl_log[experts_after]:.4f}")

    del model
    del engine

    return ppl_log


def run_worker(path, name):
    """Worker function to process a single file."""
    config = CareComV22Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_chunks = prepare_wikitext_eval_batches(tokenizer)

    with open(path, "r") as f:
        data = json.load(f)

    ppl_log = replay_merges_and_compute_ppl(
        data["trace"], name, config, tokenizer, eval_chunks
    )

    # Update the file with corrected PPL
    data["ppl"] = ppl_log
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Updated {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_path", type=str, default=None, help="Path to json file to process (internal use)")
    parser.add_argument("--worker_name", type=str, default=None, help="Name of method (internal use)")
    args = parser.parse_args()

    if args.worker_path:
        run_worker(args.worker_path, args.worker_name)
        sys.exit(0)

    config = CareComV22Config()
    
    # Process each trajectory file via subprocess
    traj_dir = config.trajectories_dir
    files_to_process = []

    # Random seeds
    for seed in config.random_seeds:
        path = os.path.join(traj_dir, f"random_seed_{seed}.json")
        if os.path.exists(path):
            files_to_process.append((path, f"Random (seed={seed})"))

    # Static
    static_path = os.path.join(traj_dir, "static_v1.json")
    if os.path.exists(static_path):
        files_to_process.append((static_path, "Static v1"))

    # Adaptive
    adaptive_path = os.path.join(traj_dir, "adaptive_v21.json")
    if os.path.exists(adaptive_path):
        files_to_process.append((adaptive_path, "Adaptive v2.1"))

    for path, name in files_to_process:
        cmd = [
            sys.executable, "-m", "experiments.care_com_v22.recompute_ppl",
            "--worker_path", path,
            "--worker_name", name
        ]
        print(f"Launching subprocess for {name}...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Worker for {name} failed with code {result.returncode}")
            sys.exit(1)

    print("\n[Done] PPL recomputed for all trajectories.")
    print("Run analysis: PYTHONPATH=. python -m experiments.care_com_v22.analysis")


if __name__ == "__main__":
    main()
