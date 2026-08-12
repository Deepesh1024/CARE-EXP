"""
EXPERIMENT 3C — PHASE 2: LONGITUDINAL ORACLE DATA GENERATION
==============================================================
Priority-ordered Oracle distance matrix computation:

  PRIORITY 1: checkpoint_100 — all 2016 pairs/layer (complete coverage)
  PRIORITY 2: checkpoint_10  — 384 manifest pairs/layer
  PRIORITY 3: checkpoint_40  — 384 manifest pairs/layer
  PRIORITY 4: checkpoint_70  — 384 manifest pairs/layer

For early checkpoints, unmeasured entries are NaN (NOT zero).
Resume-safe: skips completed checkpoint/layer combinations.
"""

import os
import sys
import gc
import csv
import json
import time
import hashlib
from datetime import datetime
import numpy as np
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

from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

from config import (
    BASE_MODEL_ID,
    N_EXPERTS,
    LAYERS,
    CHECKPOINTS,
    CHECKPOINT_PRIORITY,
    CALIBRATION_CACHE_FILE,
    CALIBRATION_METADATA_FILE,
    PAIR_MANIFEST_FILE,
    RESULTS_DIR,
    LOGS_DIR,
    CHECKPOINT_METADATA_FILE,
    RUNTIME_REPORT_FILE,
    DEVICE,
    DTYPE,
    CALIB_BATCH_SIZE,
    RANDOM_SEED,
    ensure_dirs,
)

# Import the identical rigorous Exp 1/3B oracle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment1")))
from CARE_MoE_V3_E1 import find_moe_layers, run_oracle_pair


def get_layer_block(model, layer_name, all_layers):
    """Retrieve the targeted MoE layer module block."""
    if layer_name == "first":
        return all_layers[0][1]
    elif layer_name == "middle":
        return all_layers[len(all_layers) // 2][1]
    elif layer_name == "last":
        return all_layers[-1][1]
    raise ValueError(f"Unknown layer {layer_name}")


def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_pair_manifest():
    """Load the frozen pair manifest for early checkpoints."""
    if not os.path.exists(PAIR_MANIFEST_FILE):
        raise FileNotFoundError(
            f"Pair manifest not found: {PAIR_MANIFEST_FILE}\n"
            "Run phase0_pair_manifest.py first."
        )
    with open(PAIR_MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    return manifest


def load_calibration_data():
    """
    Load the frozen calibration dataset.
    Returns stacked input_ids and attention_mask tensors.
    """
    if not os.path.exists(CALIBRATION_CACHE_FILE):
        raise FileNotFoundError(
            f"Calibration cache not found: {CALIBRATION_CACHE_FILE}\n"
            "Run phase1_calibration.py first."
        )

    print(f"[Phase 2] Loading calibration data from {CALIBRATION_CACHE_FILE}")
    sequences = torch.load(CALIBRATION_CACHE_FILE)

    # The calibration is stored as a list of dicts with "input_ids" and "attention_mask"
    input_ids = torch.stack([s["input_ids"] for s in sequences])
    attention_mask = torch.stack([s["attention_mask"] for s in sequences])

    print(f"[Phase 2] Calibration: {input_ids.shape[0]} sequences × {input_ids.shape[1]} tokens")
    return input_ids, attention_mask


def log_runtime(ckpt_name, layer, n_pairs, elapsed_time):
    csv_path = os.path.join(LOGS_DIR, "runtime.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["checkpoint", "layer", "n_pairs", "time_seconds",
                             "sec_per_pair", "timestamp"])
        sec_per_pair = elapsed_time / max(n_pairs, 1)
        writer.writerow([ckpt_name, layer, n_pairs, f"{elapsed_time:.1f}",
                         f"{sec_per_pair:.2f}", datetime.now().isoformat()])


def validate_and_save_matrix(matrix, output_path, expected_pairs, is_full):
    """
    Validate and atomically save the Oracle distance matrix.

    For full coverage (100%): all off-diagonal entries must be populated (no NaN).
    For sampled coverage: only manifest pairs are populated; rest are NaN.
    """
    assert matrix.shape == (N_EXPERTS, N_EXPERTS), f"Shape mismatch: {matrix.shape}"
    assert np.all(np.diag(matrix) == 0.0), "Diagonal must be exactly 0"

    if is_full:
        assert np.all(np.isfinite(matrix)), "Full matrix contains NaN/Inf"
        measured = np.count_nonzero(~np.isnan(matrix[np.triu_indices(N_EXPERTS, k=1)]))
        # Actually for full, there should be no NaN at all in off-diagonal
        upper_tri = matrix[np.triu_indices(N_EXPERTS, k=1)]
        assert len(upper_tri) == 2016
    else:
        # For sampled: check that exactly expected_pairs are measured
        upper_tri = matrix[np.triu_indices(N_EXPERTS, k=1)]
        measured = np.count_nonzero(~np.isnan(upper_tri))
        assert measured == expected_pairs, (
            f"Expected {expected_pairs} measured pairs, got {measured}"
        )
        # Check that measured values are finite
        measured_vals = upper_tri[~np.isnan(upper_tri)]
        assert np.all(np.isfinite(measured_vals)), "Measured values contain Inf"

    # Atomic save: write to tmp, then rename
    tmp_npy = output_path + ".tmp"
    np.save(tmp_npy, matrix)

    tmp_csv = output_path.replace(".npy", ".csv") + ".tmp"
    # Save CSV with NaN preserved
    import pandas as pd
    pd.DataFrame(matrix).to_csv(tmp_csv, index=False, header=False, na_rep="NaN")

    os.rename(tmp_npy, output_path)
    os.rename(tmp_csv, output_path.replace(".npy", ".csv"))
    return True


def process_checkpoint(ckpt_name, ckpt_info, input_ids, attention_mask, manifest):
    """Process a single training checkpoint."""
    hf_revision = ckpt_info["hf_revision"]
    is_full = ckpt_info["coverage"] == "full"

    ckpt_dir = os.path.join(RESULTS_DIR, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Check if all layers are already complete
    all_done = all(
        os.path.exists(os.path.join(ckpt_dir, layer, "COMPLETE"))
        for layer in LAYERS
    )
    if all_done:
        print(f"[Phase 2] {ckpt_name} fully COMPLETE. Skipping.")
        return

    print(f"\n{'='*60}")
    print(f"[Phase 2] Loading: {ckpt_name} (Revision: {hf_revision})")
    print(f"  Coverage: {'FULL (2016 pairs/layer)' if is_full else 'SAMPLED (384 pairs/layer)'}")
    print(f"  Target: {ckpt_info['target_pct']}% | Actual: {ckpt_info['actual_pct']}%")
    print(f"{'='*60}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            revision=hf_revision,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        ).eval()
    except Exception as e:
        print(f"FAILED to load model {hf_revision}: {e}")
        return

    # Get calibration hash for metadata
    calib_hash = get_sha256(CALIBRATION_CACHE_FILE)

    moe_layers = find_moe_layers(model)
    print(f"[Phase 2] Found {len(moe_layers)} MoE layers")

    # Save per-checkpoint metadata
    ckpt_meta = {
        "checkpoint_name": ckpt_name,
        "hf_revision": hf_revision,
        "actual_step": ckpt_info["actual_step"],
        "actual_tokens_B": ckpt_info["actual_tokens_B"],
        "actual_pct": ckpt_info["actual_pct"],
        "target_pct": ckpt_info["target_pct"],
        "coverage": ckpt_info["coverage"],
        "calibration_sha256": calib_hash,
        "n_moe_layers": len(moe_layers),
        "n_experts": N_EXPERTS,
        "device": str(DEVICE),
        "timestamp_start": datetime.now().isoformat(),
    }
    meta_path = os.path.join(ckpt_dir, "checkpoint_info.json")
    with open(meta_path, "w") as f:
        json.dump(ckpt_meta, f, indent=2)

    for layer in LAYERS:
        layer_dir = os.path.join(ckpt_dir, layer)
        os.makedirs(layer_dir, exist_ok=True)
        complete_marker = os.path.join(layer_dir, "COMPLETE")

        if os.path.exists(complete_marker):
            print(f"[Phase 2] {ckpt_name}/{layer} already COMPLETE. Skipping.")
            continue

        print(f"\n  [Phase 2] Processing: {ckpt_name} / {layer}")
        start_time = time.time()

        target_block = get_layer_block(model, layer, moe_layers)

        # Determine pairs to compute
        if is_full:
            pairs_to_run = []
            for i in range(N_EXPERTS):
                for j in range(i + 1, N_EXPERTS):
                    pairs_to_run.append((i, j))
            expected_pairs = 2016
        else:
            # Use the manifest pairs for this layer
            manifest_pairs = manifest["layers"][layer]["pairs"]
            pairs_to_run = [(p[0], p[1]) for p in manifest_pairs]
            expected_pairs = len(pairs_to_run)

        # Initialize matrix: NaN for unmeasured, 0 on diagonal
        matrix = np.full((N_EXPERTS, N_EXPERTS), np.nan, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)

        # Check for partial progress (resume support)
        partial_path = os.path.join(layer_dir, "partial_matrix.npy")
        pairs_already_done = set()
        if os.path.exists(partial_path):
            matrix = np.load(partial_path)
            # Identify which pairs are already measured
            for i, j in pairs_to_run:
                if not np.isnan(matrix[i, j]):
                    pairs_already_done.add((i, j))
            print(f"  Resuming: {len(pairs_already_done)}/{len(pairs_to_run)} pairs already done")

        pairs_remaining = [(i, j) for (i, j) in pairs_to_run if (i, j) not in pairs_already_done]

        pbar = tqdm(pairs_remaining, desc=f"Oracle {layer}",
                    initial=len(pairs_already_done), total=len(pairs_to_run))

        pairs_computed = 0
        save_interval = 50  # Save partial progress every 50 pairs

        for (i, j) in pbar:
            oracle_result = run_oracle_pair(
                model=model,
                moe_block=target_block,
                i=i,
                j=j,
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_size=CALIB_BATCH_SIZE,
                top_k=getattr(model.config, "num_experts_per_tok", 8),
            )
            kl_div = oracle_result["Oracle_KL"]

            matrix[i, j] = kl_div
            matrix[j, i] = kl_div
            pairs_computed += 1
            pbar.set_postfix({"KL": f"{kl_div:.6f}"})

            # Periodic partial save for resume safety
            if pairs_computed % save_interval == 0:
                np.save(partial_path, matrix)

        # Final partial save before validation
        np.save(partial_path, matrix)

        # Validate and atomically save
        out_path = os.path.join(layer_dir, "oracle_distance.npy")
        try:
            validate_and_save_matrix(matrix, out_path, expected_pairs, is_full)
        except AssertionError as e:
            print(f"  VALIDATION FAILED for {ckpt_name}/{layer}: {e}")
            continue

        elapsed = time.time() - start_time

        # Write COMPLETE marker with metadata
        with open(complete_marker, "w") as f:
            json.dump({
                "completed_at": datetime.now().isoformat(),
                "pairs_computed": pairs_computed + len(pairs_already_done),
                "expected_pairs": expected_pairs,
                "elapsed_seconds": elapsed,
                "sec_per_pair": elapsed / max(pairs_computed, 1),
                "calibration_sha256": calib_hash,
            }, f, indent=2)

        log_runtime(ckpt_name, layer, pairs_computed + len(pairs_already_done), elapsed)
        print(f"  [Phase 2] {ckpt_name}/{layer} COMPLETE. "
              f"{pairs_computed + len(pairs_already_done)} pairs in {elapsed:.0f}s")

        # Clean up partial file after successful completion
        if os.path.exists(partial_path):
            os.remove(partial_path)

        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    ensure_dirs()

    print("=" * 70)
    print("EXPERIMENT 3C — PHASE 2: LONGITUDINAL ORACLE GENERATION")
    print("=" * 70)

    # Load calibration data
    input_ids, attention_mask = load_calibration_data()

    # Load pair manifest
    manifest = load_pair_manifest()
    print(f"[Phase 2] Loaded pair manifest: {manifest['pairs_per_layer']} pairs/layer")

    # Process checkpoints in PRIORITY ORDER
    print(f"\n[Phase 2] Execution priority: {CHECKPOINT_PRIORITY}")

    for ckpt_name in CHECKPOINT_PRIORITY:
        ckpt_info = CHECKPOINTS[ckpt_name]
        process_checkpoint(ckpt_name, ckpt_info, input_ids, attention_mask, manifest)

    # Save aggregated checkpoint metadata
    all_meta = {}
    for ckpt_name in CHECKPOINTS:
        meta_path = os.path.join(RESULTS_DIR, ckpt_name, "checkpoint_info.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                all_meta[ckpt_name] = json.load(f)
    with open(CHECKPOINT_METADATA_FILE, "w") as f:
        json.dump(all_meta, f, indent=2)

    # Save runtime report
    runtime_csv = os.path.join(LOGS_DIR, "runtime.csv")
    runtime_data = {"generated_at": datetime.now().isoformat(), "entries": []}
    if os.path.exists(runtime_csv):
        import csv as csv_mod
        with open(runtime_csv, "r") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                runtime_data["entries"].append(row)
    with open(RUNTIME_REPORT_FILE, "w") as f:
        json.dump(runtime_data, f, indent=2)

    print(f"\n[Phase 2] ALL CHECKPOINTS PROCESSED.")
    print(f"[Phase 2] Metadata: {CHECKPOINT_METADATA_FILE}")
    print(f"[Phase 2] Runtime: {RUNTIME_REPORT_FILE}")


if __name__ == "__main__":
    main()
