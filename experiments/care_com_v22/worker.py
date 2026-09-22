"""
CARE-COM v2.2 Worker — runs a single baseline in an isolated process.
Called by driver.py via subprocess to guarantee complete VRAM cleanup between runs.

Usage:
    PYTHONPATH=. python -m experiments.care_com_v22.worker --method random --seed 42
    PYTHONPATH=. python -m experiments.care_com_v22.worker --method static
    PYTHONPATH=. python -m experiments.care_com_v22.worker --method adaptive
"""
import os
import sys
import json
import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment7b")))
from utils.evaluation import prepare_wikitext_eval_batches

from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.baselines import run_random_baseline, run_static_baseline
from experiments.care_com_v22.adaptive import run_adaptive_baseline


def load_calibration_data():
    """Loads the CARE capability token vectors (parquet DataFrame)."""
    token_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "results", "exp6c", "token_vectors",
        "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    )
    if not os.path.exists(token_path):
        token_path = "results/exp6c/token_vectors/EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"

    print(f"Loading calibration data from {token_path}...")
    df_tokens = pd.read_parquet(token_path)
    return df_tokens


def load_model(config):
    """Loads a fresh uncompressed model."""
    print(f"Loading {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=config.device
    )
    model.eval()
    return model


def run_worker(method, seed=None):
    config = CareComV22Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Eval chunks: list of dicts with input_ids & attention_mask (from experiment7b)
    eval_chunks = prepare_wikitext_eval_batches(tokenizer)

    # Calibration data: pandas DataFrame with axis_idx, input_ids, attention_mask
    df_tokens = load_calibration_data()

    model = load_model(config)
    engine = PhysicalMergeEngine(model)

    if method == "random":
        assert seed is not None, "Random baseline requires --seed"
        print(f"\n{'='*60}")
        print(f"  CARE-COM v2.2 Worker: Random Baseline (seed={seed})")
        print(f"{'='*60}")
        trace, ppl_log = run_random_baseline(model, engine, eval_chunks, config, seed=seed)
        out_path = os.path.join(config.trajectories_dir, f"random_seed_{seed}.json")

    elif method == "static":
        print(f"\n{'='*60}")
        print(f"  CARE-COM v2.2 Worker: Static v1 Baseline")
        print(f"{'='*60}")
        trace, ppl_log = run_static_baseline(model, engine, df_tokens, eval_chunks, config)
        out_path = os.path.join(config.trajectories_dir, "static_v1.json")

    elif method == "adaptive":
        print(f"\n{'='*60}")
        print(f"  CARE-COM v2.2 Worker: Adaptive v2.1 Baseline")
        print(f"{'='*60}")
        trace, ppl_log = run_adaptive_baseline(model, engine, df_tokens, eval_chunks, config)
        out_path = os.path.join(config.trajectories_dir, "adaptive_v21.json")

    else:
        raise ValueError(f"Unknown method: {method}")

    with open(out_path, "w") as f:
        json.dump({"trace": trace, "ppl": ppl_log}, f, indent=2)

    print(f"\n[Worker] Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE-COM v2.2 isolated worker")
    parser.add_argument("--method", required=True, choices=["random", "static", "adaptive"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_worker(args.method, args.seed)
