"""
CARE-COM v2.2 Worker — runs a single baseline in an isolated process.
Called by driver.py via subprocess to guarantee complete VRAM cleanup between runs.

Usage:
    python -m experiments.care_com_v22.worker --method random --seed 42
    python -m experiments.care_com_v22.worker --method static
    python -m experiments.care_com_v22.worker --method adaptive
"""
import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.baselines import run_random_baseline, run_static_baseline
from experiments.care_com_v22.adaptive import run_adaptive_baseline


def prepare_data(tokenizer, config, max_length=1024):
    """Loads wikitext for evaluation."""
    print("Loading WikiText...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    num_eval_chunks = config.max_ppl_batches * config.ppl_batch_size
    chunks = []
    for i in range(0, min(len(tokens) - max_length, num_eval_chunks * max_length), max_length):
        chunk = tokens[i : i + max_length]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk)
        })

    df_tokens = tokens[:4096].unsqueeze(0)
    return chunks, df_tokens


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
    eval_chunks, df_tokens = prepare_data(tokenizer, config)
    df_tokens = df_tokens.to(config.device)

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
