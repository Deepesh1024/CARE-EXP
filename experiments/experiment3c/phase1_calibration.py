"""
EXPERIMENT 3C — PHASE 1: CALIBRATION CORPUS GENERATION
========================================================
Generates the frozen calibration dataset using EXACTLY the validated
Experiment 1/3B methodology:

  Dataset:   Salesforce/wikitext (wikitext-2-raw-v1), split=train
  Seed:      42
  Seq len:   512 (truncation=True, padding="max_length")
  Filter:    attention_mask.sum() >= 8
  Count:     98 sequences
  Format:    List of {"input_ids": Tensor, "attention_mask": Tensor}

This produces the same format that run_oracle_pair() in CARE_MoE_V3_E1.py expects.
"""

import os
import json
import torch
import hashlib
import random

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

from transformers import AutoTokenizer
from datasets import load_dataset

from config import (
    BASE_MODEL_ID,
    CALIBRATION_DATASET,
    CALIBRATION_SUBSET,
    CALIBRATION_SPLIT,
    CALIBRATION_SEQ_LEN,
    CALIBRATION_N_SEQUENCES,
    CALIBRATION_CACHE_FILE,
    CALIBRATION_METADATA_FILE,
    RANDOM_SEED,
    ensure_dirs,
)


def set_seed():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    ensure_dirs()
    set_seed()

    print("=" * 70)
    print("EXPERIMENT 3C — PHASE 1: CALIBRATION CORPUS GENERATION")
    print("=" * 70)

    # Check for existing valid calibration
    if os.path.exists(CALIBRATION_CACHE_FILE) and os.path.exists(CALIBRATION_METADATA_FILE):
        print(f"[Phase 1] Calibration data already exists at {CALIBRATION_CACHE_FILE}")
        with open(CALIBRATION_METADATA_FILE, "r") as f:
            meta = json.load(f)
        print(f"  Existing hash: {meta['calibration_sha256']}")
        print(f"  Sequences: {meta['num_sequences']}")
        print(f"  Dataset: {meta['dataset']}")
        if (meta["num_sequences"] == CALIBRATION_N_SEQUENCES and
                meta["dataset"] == CALIBRATION_DATASET):
            print("[Phase 1] Calibration matches spec. Skipping generation.")
            return
        print("[Phase 1] Calibration does not match spec. Re-generating...")

    # Load tokenizer
    print(f"\n[Phase 1] Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load wikitext dataset (matching Experiment 1 exactly)
    print(f"[Phase 1] Loading dataset: {CALIBRATION_DATASET} ({CALIBRATION_SUBSET}), split={CALIBRATION_SPLIT}")
    raw = load_dataset(CALIBRATION_DATASET, CALIBRATION_SUBSET, split=CALIBRATION_SPLIT)

    # Filter empty texts, shuffle deterministically (matching Experiment 1)
    texts = [t for t in raw["text"] if len(t.strip()) > 0]
    random.Random(RANDOM_SEED).shuffle(texts)
    print(f"[Phase 1] Found {len(texts)} non-empty texts after filtering")

    # Tokenize with padding="max_length" (matching Experiment 1 exactly)
    all_sequences = []
    for t in texts:
        if len(all_sequences) >= CALIBRATION_N_SEQUENCES:
            break
        enc = tokenizer(
            t,
            truncation=True,
            max_length=CALIBRATION_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        # Filter sequences with >= 8 real tokens (matching Experiment 1)
        if enc["attention_mask"].sum().item() >= 8:
            all_sequences.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    print(f"[Phase 1] Collected {len(all_sequences)} sequences of length {CALIBRATION_SEQ_LEN}")

    if len(all_sequences) < CALIBRATION_N_SEQUENCES:
        print(f"[Phase 1] WARNING: Only collected {len(all_sequences)} sequences "
              f"(target was {CALIBRATION_N_SEQUENCES})")

    # Compute actual token counts
    total_real_tokens = sum(seq["attention_mask"].sum().item() for seq in all_sequences)
    total_padded_tokens = len(all_sequences) * CALIBRATION_SEQ_LEN

    print(f"[Phase 1] Real tokens: {int(total_real_tokens)}")
    print(f"[Phase 1] Padded tokens: {total_padded_tokens}")

    # Save calibration data
    print(f"[Phase 1] Saving to {CALIBRATION_CACHE_FILE}")
    os.makedirs(os.path.dirname(CALIBRATION_CACHE_FILE), exist_ok=True)
    torch.save(all_sequences, CALIBRATION_CACHE_FILE)

    # Hash the file
    file_hash = get_sha256(CALIBRATION_CACHE_FILE)

    # Save metadata
    real_token_counts = [int(seq["attention_mask"].sum().item()) for seq in all_sequences]
    metadata = {
        "dataset": CALIBRATION_DATASET,
        "subset": CALIBRATION_SUBSET,
        "split": CALIBRATION_SPLIT,
        "tokenizer": BASE_MODEL_ID,
        "num_sequences": len(all_sequences),
        "seq_length": CALIBRATION_SEQ_LEN,
        "padding": "max_length",
        "total_real_tokens": int(total_real_tokens),
        "total_padded_tokens": total_padded_tokens,
        "min_real_tokens": min(real_token_counts),
        "max_real_tokens": max(real_token_counts),
        "mean_real_tokens": sum(real_token_counts) / len(real_token_counts),
        "calibration_sha256": file_hash,
        "seed": RANDOM_SEED,
        "methodology": "Identical to Experiment 1/3B: Salesforce/wikitext, "
                        "truncation=True, max_length=512, padding=max_length, "
                        "filter attention_mask.sum()>=8",
    }

    with open(CALIBRATION_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[Phase 1] Saved metadata to {CALIBRATION_METADATA_FILE}")
    print(f"  -> SHA256: {file_hash}")
    print(f"  -> Sequences: {len(all_sequences)}")
    print(f"  -> Real tokens: {int(total_real_tokens)}")
    print("[Phase 1] CALIBRATION GENERATION COMPLETE.")


if __name__ == "__main__":
    main()
