"""
EXPERIMENT 7A - PHASE 1: DATASET PARTITIONING
======================================================
Loads ARC-Challenge dataset and reproducibly creates four partitions:
D_screen, D_proxy, D_validation, D_final.
"""

import os
import json
import torch
import random
from datasets import load_dataset

from config import (
    DATASET_NAME, DATASET_SUBSET, RANDOM_SEED,
    SPLIT_SIZES, DATA_DIR,
    ensure_dirs, mark_task, is_task_completed
)

def set_seed():
    random.seed(RANDOM_SEED)

def generate_partitions():
    task_id = "phase1_dataset"
    if is_task_completed(task_id):
        print("[Phase 1] Dataset partitions already exist. Skipping.")
        return

    ensure_dirs()
    set_seed()

    print(f"[Phase 1] Loading dataset: {DATASET_NAME} ({DATASET_SUBSET})")
    try:
        ds = load_dataset(DATASET_NAME, DATASET_SUBSET)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Pool all examples from train, validation, and test
    all_examples = []
    for split in ds.keys():
        for item in ds[split]:
            all_examples.append(item)

    total_examples = len(all_examples)
    print(f"[Phase 1] Total labeled examples available: {total_examples}")
    
    # Verify we have enough examples
    required = sum(SPLIT_SIZES.values())
    assert total_examples >= required, f"Not enough examples. Have {total_examples}, need {required}"

    # Shuffle deterministically
    random.shuffle(all_examples)

    # Partition
    partitions = {}
    current_idx = 0
    for split_name, size in SPLIT_SIZES.items():
        partitions[split_name] = all_examples[current_idx : current_idx + size]
        current_idx += size
        print(f"[Phase 1] Partition {split_name}: {size} examples")

    # Save to disk
    manifest = {
        "dataset_name": DATASET_NAME,
        "subset": DATASET_SUBSET,
        "seed": RANDOM_SEED,
        "splits": {}
    }

    for split_name, data in partitions.items():
        out_path = os.path.join(DATA_DIR, f"{split_name}.pt")
        torch.save(data, out_path)
        
        # Save manifest info
        manifest["splits"][split_name] = {
            "size": len(data),
            "file": os.path.basename(out_path),
            "sample_ids": [item["id"] for item in data]
        }

    manifest_path = os.path.join(DATA_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"[Phase 1] Saved partitions and manifest to {DATA_DIR}")
    mark_task(task_id, "completed")

if __name__ == "__main__":
    generate_partitions()
