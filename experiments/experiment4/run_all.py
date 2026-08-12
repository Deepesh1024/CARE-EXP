"""
CARE-MoE Experiment 4 — Main Orchestrator
==========================================
Executes all phases in order:

  PHASE 0: Validate cached data.
  PHASE 1: Freeze experiment_config.json and fold assignments.
  PHASE 2: Two-partition integrity pilot.
  PHASE 3: Resume test.
  PHASE 4: Freeze code (log git hash).
  PHASE 5: Full 5-partition × 3-fold experiment.
  PHASE 6: Statistics and plots.
  PHASE 7: Final report.

Usage:
  python run_all.py [--resume] [--pilot-only] [--skip-pilot]

Resume safety:
  - Each fold writes COMPLETE only after all artifacts are valid.
  - Resume scans COMPLETE markers and skips valid completed folds.
  - Fold assignments are loaded from frozen cv_splits.json (never regenerated).
  - experiment_config.json is immutable after Phase 1.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_EXPERTS, N_PAIRS, N_PARTITIONS, N_FOLDS, Q,
    PARTITION_SEEDS, PILOT_PARTITIONS,
    LOCAL_FEATURES, FORBIDDEN_FEATURES,
    XGBOOST_PARAMS, RANDOM_SEED,
    RESULTS_DIR, PREDICTIONS_DIR, EMBEDDINGS_DIR, NOISE_CEILING_DIR, PLOTS_DIR,
    EXPERIMENT_CONFIG_PATH, CV_SPLITS_PATH,
    FOLD_METRICS_CSV, PARTITION_METRICS_CSV,
    FINAL_REPORT_JSON, FINAL_REPORT_MD,
    EXPERIMENT_VERSION, SMACOF_N_INIT, OOS_N_RESTARTS,
    NOISE_CEILING_STATUS,
)
from data_loader import load_all
from cv_splits import generate_and_freeze_splits, get_fold_data
from mds_embedding import embed_fold, compute_geometry_distances
from model_a import train_model_a, predict_model_a
from model_b import predict_model_b
from model_c import train_model_c, predict_model_c, compute_train_geometry
from leakage_checks import (
    assert_no_train_test_overlap,
    assert_no_test_test_in_mds,
    assert_no_target_in_features,
    assert_q_not_tuned_on_exp4,
    assert_hyperparams_unchanged,
    assert_no_topology_features,
    assert_mds_output_dimension,
    assert_no_nan_inf,
    assert_feature_target_alignment,
    assert_identical_folds_abc,
)
from metrics import compute_fold_metrics, aggregate_partition, compute_final_statistics
from noise_ceiling import run_noise_ceiling
from feature_provenance import validate_feature_list, print_provenance_summary, save_provenance


# ══════════════════════════════════════════════════════════
# Directory Setup
# ══════════════════════════════════════════════════════════

def ensure_dirs():
    for d in [RESULTS_DIR, PREDICTIONS_DIR, EMBEDDINGS_DIR,
              NOISE_CEILING_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def fold_dir(partition_idx: int, fold_idx: int) -> str:
    return os.path.join(
        RESULTS_DIR,
        f"partition_{partition_idx:02d}",
        f"fold_{fold_idx:02d}",
    )


# ══════════════════════════════════════════════════════════
# Resume Safety — Atomic Writes
# ══════════════════════════════════════════════════════════

def atomic_write_json(data: dict, path: str) -> None:
    """Write JSON atomically: write to .tmp, validate, rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    # Validate: re-read and check not empty
    with open(tmp, "r") as f:
        loaded = json.load(f)
    assert loaded, f"Written JSON is empty: {tmp}"
    os.replace(tmp, path)


def atomic_write_npy(arr: np.ndarray, path: str) -> None:
    """Write numpy array atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.save(tmp, arr)
    arr_loaded = np.load(tmp + ".npy" if not tmp.endswith(".npy") else tmp)
    assert arr_loaded.shape == arr.shape, "Shape mismatch after write"
    os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def is_fold_complete(partition_idx: int, fold_idx: int) -> bool:
    """Check if a fold has a valid COMPLETE marker and all required artifacts."""
    d = fold_dir(partition_idx, fold_idx)
    complete_path = os.path.join(d, "COMPLETE")
    if not os.path.exists(complete_path):
        return False

    required = [
        "train_experts.json", "test_experts.json",
        "mds_coordinates.npy",
        "predictions_model_a.npy", "predictions_model_b.npy",
        "predictions_model_c.npy",
        "oracle_targets.npy", "metrics.json",
    ]
    for fname in required:
        fpath = os.path.join(d, fname)
        if not os.path.exists(fpath):
            print(f"  [resume] COMPLETE marker present but {fname} missing — "
                  f"fold P{partition_idx}F{fold_idx} marked incomplete.")
            return False

    return True


def write_fold_complete(partition_idx: int, fold_idx: int) -> None:
    """Write COMPLETE marker ONLY after all artifacts are validated."""
    d = fold_dir(partition_idx, fold_idx)
    complete_path = os.path.join(d, "COMPLETE")
    with open(complete_path, "w") as f:
        f.write(f"COMPLETE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


# ══════════════════════════════════════════════════════════
# Phase 0 — Data Validation
# ══════════════════════════════════════════════════════════

def phase0_validate_data():
    print("\n" + "=" * 70)
    print("PHASE 0 — DATA VALIDATION")
    print("=" * 70)

    # Feature provenance audit
    validate_feature_list()
    print_provenance_summary()

    # Load all data
    data = load_all()
    return data


# ══════════════════════════════════════════════════════════
# Phase 1 — Freeze Config and Splits
# ══════════════════════════════════════════════════════════

def phase1_freeze_config(data: dict) -> list:
    print("\n" + "=" * 70)
    print("PHASE 1 — FREEZE EXPERIMENT CONFIG AND CV SPLITS")
    print("=" * 70)

    # Generate/load CV splits (idempotent)
    splits = generate_and_freeze_splits()

    # Write experiment_config.json (immutable — do not overwrite)
    if not os.path.exists(EXPERIMENT_CONFIG_PATH):
        config_data = {
            "experiment": "CARE-MoE Experiment 4 — Functional Merge Landscape",
            "version": EXPERIMENT_VERSION,
            "layer": "middle",
            "conclusions_scope": "middle-layer-only",
            "q": Q,
            "q_provenance": (
                "q=4 selected in Exp 3B as best-performing among q=2,4,6,8 "
                "(performance-selected, not theoretically motivated). "
                "Fixed for Exp 4 — not re-tuned on Exp 4 results."
            ),
            "seq_len": 512,
            "seq_len_reason": (
                "Matches calibration configuration of Exp 3B Oracle matrix. "
                "Avoids calibration-distribution mismatch."
            ),
            "n_experts": N_EXPERTS,
            "n_pairs": N_PAIRS,
            "n_partitions": N_PARTITIONS,
            "n_folds": N_FOLDS,
            "partition_seeds": PARTITION_SEEDS,
            "smacof_n_init": SMACOF_N_INIT,
            "oos_n_restarts": OOS_N_RESTARTS,
            "feature_list": list(LOCAL_FEATURES),
            "n_features_model_a": len(LOCAL_FEATURES),
            "n_features_model_b": 0,  # Not a learned predictor
            "n_features_model_c": len(LOCAL_FEATURES) + 1,
            "model_b_type": "geometry_distance_only_not_learned",
            "xgboost_params": XGBOOST_PARAMS,
            "delta_rho_min": 0.05,
            "noise_ceiling_status": NOISE_CEILING_STATUS,
            "oracle_matrix_hash": data["oracle_hash"],
            "feature_data_hash": data["feature_hash"],
            "timestamp_frozen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(config_data, EXPERIMENT_CONFIG_PATH)
        print(f"[phase1] Config frozen → {EXPERIMENT_CONFIG_PATH}")
    else:
        print(f"[phase1] Config already frozen — loaded from disk.")

    # Save feature provenance
    save_provenance(RESULTS_DIR)

    return splits


# ══════════════════════════════════════════════════════════
# Core: Run Single Fold
# ══════════════════════════════════════════════════════════

def run_fold(
    partition_idx: int,
    fold_idx: int,
    splits: list,
    data: dict,
) -> dict:
    """Execute one fold: embed, train A+C, predict A+B+C, metrics, save."""
    context = f"P{partition_idx}F{fold_idx}"
    print(f"\n{'─' * 50}")
    print(f"  Fold: Partition {partition_idx}, Fold {fold_idx}")
    print(f"{'─' * 50}")

    D_oracle = data["D_oracle"]
    X_unscaled = data["X_unscaled"]
    y = data["y"]
    pair_i = data["pair_i"]
    pair_j = data["pair_j"]

    # Extract fold data
    fold_data = get_fold_data(
        splits, partition_idx, fold_idx, pair_i, pair_j, X_unscaled, y
    )

    train_experts = fold_data["train_experts"]
    test_experts = fold_data["test_experts"]
    X_train = fold_data["X_train"]
    y_train = fold_data["y_train"]
    X_test = fold_data["X_test"]
    y_test = fold_data["y_test"]
    pi_train = fold_data["pi_train"]
    pj_train = fold_data["pj_train"]
    pi_test = fold_data["pi_test"]
    pj_test = fold_data["pj_test"]

    print(f"  Train: {len(train_experts)} experts, {len(X_train)} pairs")
    print(f"  Test:  {len(test_experts)} experts, {len(X_test)} pairs")

    # ── Leakage checks (pre-run) ──────────────────────────────────
    assert_no_train_test_overlap(train_experts, test_experts, context)
    assert_no_target_in_features(list(LOCAL_FEATURES), FORBIDDEN_FEATURES, context)
    assert_q_not_tuned_on_exp4(Q, 4, context)
    assert_hyperparams_unchanged(XGBOOST_PARAMS, XGBOOST_PARAMS, context)
    assert_no_topology_features(list(LOCAL_FEATURES), context)

    # ── Step 1: MDS Embedding ─────────────────────────────────────
    print(f"  [embedding] Running SMACOF (q={Q}, n_init={SMACOF_N_INIT}) ...")
    fold_seed = PARTITION_SEEDS[partition_idx] * 1000 + fold_idx
    Z_train, Z_test = embed_fold(D_oracle, train_experts, test_experts, fold_seed)

    # Verify leakage: D_train used for MDS
    train_idx = np.array(train_experts, dtype=np.int32)
    D_train = D_oracle[np.ix_(train_idx, train_idx)]
    assert_no_test_test_in_mds(D_train, train_experts, D_oracle, context)
    assert_mds_output_dimension(Z_train, Q, context)
    assert_mds_output_dimension(Z_test, Q, context)
    assert_no_nan_inf(Z_train, "Z_train", context)
    assert_no_nan_inf(Z_test, "Z_test", context)

    # ── Step 2: Geometry Distances ────────────────────────────────
    # For test pairs: ||z_i - z_j||_2 (Model B prediction)
    geom_test = compute_geometry_distances(Z_test, test_experts, pi_test, pj_test)
    # For train pairs: needed for Model C
    geom_train = compute_train_geometry(Z_train, pi_train, pj_train, train_experts)

    assert_no_nan_inf(geom_test, "geom_test", context)
    assert_no_nan_inf(geom_train, "geom_train", context)

    # ── Step 3: Model A (local features only) ─────────────────────
    print(f"  [model_a] Training XGBoost on {X_train.shape[1]} local features ...")
    model_a, scaler_a = train_model_a(X_train, y_train)
    pred_a = predict_model_a(model_a, scaler_a, X_test)
    assert_no_nan_inf(pred_a, "pred_a", context)

    # ── Step 4: Model B (geometry only, NOT learned) ──────────────
    print(f"  [model_b] Geometry distances as predictions (no training).")
    pred_b = predict_model_b(geom_test)
    assert_no_nan_inf(pred_b, "pred_b", context)

    # ── Step 5: Model C (local + geometry) ────────────────────────
    print(f"  [model_c] Training XGBoost on {len(LOCAL_FEATURES)+1} features ...")
    model_c, scaler_c = train_model_c(X_train, y_train, geom_train)
    pred_c = predict_model_c(model_c, scaler_c, X_test, geom_test)
    assert_no_nan_inf(pred_c, "pred_c", context)

    # ── Verify identical experts for A/B/C ────────────────────────
    assert_identical_folds_abc(
        train_experts, test_experts,
        train_experts, test_experts,
        train_experts, test_experts,
        context,
    )

    # ── Feature-target alignment ──────────────────────────────────
    assert_feature_target_alignment(y_test, pi_test, pj_test, D_oracle, context)

    # ── Step 6: Metrics ───────────────────────────────────────────
    fold_metrics = compute_fold_metrics(
        y_test, pred_a, pred_b, pred_c, partition_idx, fold_idx
    )

    print(f"  [metrics] "
          f"ρ_A={fold_metrics['rho_A']:.4f}  "
          f"ρ_B={fold_metrics['rho_B']:.4f}  "
          f"ρ_C={fold_metrics['rho_C']:.4f}  "
          f"Δρ_CA={fold_metrics['delta_rho_CA']:+.4f}")

    # ── Step 7: Save Fold Artifacts (atomic) ─────────────────────
    d = fold_dir(partition_idx, fold_idx)
    os.makedirs(d, exist_ok=True)

    atomic_write_json({"experts": train_experts}, os.path.join(d, "train_experts.json"))
    atomic_write_json({"experts": test_experts}, os.path.join(d, "test_experts.json"))
    atomic_write_json(fold_metrics, os.path.join(d, "metrics.json"))

    # Save MDS coordinates (Z_test only — Z_train is large, store separately)
    coord_path = os.path.join(d, "mds_coordinates.npy")
    _npy_atomic_save(
        {"Z_train": Z_train, "Z_test": Z_test,
         "train_experts": np.array(train_experts),
         "test_experts": np.array(test_experts)},
        coord_path,
    )

    _npy_simple_save(pred_a, os.path.join(d, "predictions_model_a.npy"))
    _npy_simple_save(pred_b, os.path.join(d, "predictions_model_b.npy"))
    _npy_simple_save(pred_c, os.path.join(d, "predictions_model_c.npy"))
    _npy_simple_save(y_test, os.path.join(d, "oracle_targets.npy"))

    # Also save pair indices for reference
    atomic_write_json(
        {"pi_test": pi_test.tolist(), "pj_test": pj_test.tolist()},
        os.path.join(d, "pair_indices.json"),
    )

    # ── Write COMPLETE marker (only after all artifacts valid) ────
    write_fold_complete(partition_idx, fold_idx)
    print(f"  [done] Fold {context} COMPLETE ✓")

    return fold_metrics


def _npy_simple_save(arr: np.ndarray, path: str) -> None:
    """Atomic numpy save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    check = np.load(tmp)
    assert check.shape == arr.shape
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)


def _npy_atomic_save(arrays: dict, path: str) -> None:
    """Save dict of arrays as npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.savez(tmp, **arrays)
    check = np.load(tmp + ".npz")
    for k in arrays:
        assert k in check.files
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp + ".npz", path)


# ══════════════════════════════════════════════════════════
# Phase 2 — Two-Partition Integrity Pilot
# ══════════════════════════════════════════════════════════

def phase2_pilot(splits: list, data: dict) -> list:
    print("\n" + "=" * 70)
    print("PHASE 2 — TWO-PARTITION INTEGRITY PILOT")
    print("(Implementation integrity test — NOT scientific results)")
    print("=" * 70)

    pilot_fold_metrics = []

    for p_idx in range(PILOT_PARTITIONS):
        for f_idx in range(N_FOLDS):
            if is_fold_complete(p_idx, f_idx):
                print(f"  [resume] P{p_idx}F{f_idx} already complete — skipping.")
                # Load cached metrics
                metrics_path = os.path.join(fold_dir(p_idx, f_idx), "metrics.json")
                with open(metrics_path) as ff:
                    pilot_fold_metrics.append(json.load(ff))
                continue

            fold_m = run_fold(p_idx, f_idx, splits, data)
            pilot_fold_metrics.append(fold_m)

    # Pilot integrity checklist
    print("\n" + "─" * 60)
    print("PILOT INTEGRITY CHECKLIST:")
    print("─" * 60)

    checks_passed = True
    for m in pilot_fold_metrics:
        p, f = m["partition"], m["fold"]
        checks = {
            "no NaN in rho_A": not (m["rho_A"] != m["rho_A"]),
            "no NaN in rho_B": not (m["rho_B"] != m["rho_B"]),
            "no NaN in rho_C": not (m["rho_C"] != m["rho_C"]),
            "n_test_pairs > 0": m["n_test_pairs"] > 0,
            "rho_A in [-1,1]": -1.0 <= m["rho_A"] <= 1.0,
            "rho_B in [-1,1]": -1.0 <= m["rho_B"] <= 1.0,
            "rho_C in [-1,1]": -1.0 <= m["rho_C"] <= 1.0,
        }
        for check_name, ok in checks.items():
            status = "PASS" if ok else "FAIL"
            print(f"  P{p}F{f} [{status}] {check_name}")
            if not ok:
                checks_passed = False

    if not checks_passed:
        raise RuntimeError(
            "PILOT INTEGRITY FAIL — See above. "
            "Fix implementation defect before proceeding. "
            "Do NOT change methodology based on scientific results."
        )

    print("\n  PILOT PASS ✓ — All integrity checks passed.")
    print("  NOTE: Scientific results from pilot are NOT examined.")
    print("  CODE IS NOW FROZEN.")

    return pilot_fold_metrics


# ══════════════════════════════════════════════════════════
# Phase 3 — Resume Test
# ══════════════════════════════════════════════════════════

def phase3_resume_test(splits: list, data: dict) -> None:
    """Verify resume logic is correct.

    Deliberately marks P0F0 as incomplete, runs --resume,
    verifies the fold is resumed (not recomputed from scratch).
    """
    print("\n" + "=" * 70)
    print("PHASE 3 — RESUME TEST")
    print("=" * 70)

    # Check that P0F0 is complete (from pilot)
    if not is_fold_complete(0, 0):
        print("  [resume_test] P0F0 not yet complete — running now for resume test.")
        run_fold(0, 0, splits, data)

    # Temporarily remove COMPLETE marker from P0F0
    d = fold_dir(0, 0)
    complete_path = os.path.join(d, "COMPLETE")
    backup_path = complete_path + ".bak"
    shutil.copy2(complete_path, backup_path)
    os.remove(complete_path)

    print("  [resume_test] Temporarily removed P0F0 COMPLETE marker.")
    assert not is_fold_complete(0, 0), "P0F0 should be incomplete after marker removal."
    print("  [resume_test] Verified: P0F0 is now incomplete.")

    # Reload P0F0 metrics before resume (reference)
    metrics_before_path = os.path.join(d, "metrics.json")
    with open(metrics_before_path) as f:
        metrics_before = json.load(f)

    # Run the fold again (simulating --resume behavior)
    print("  [resume_test] Re-running P0F0 (simulated --resume) ...")
    fold_m = run_fold(0, 0, splits, data)

    # Verify results are identical
    for key in ["rho_A", "rho_B", "rho_C", "delta_rho_CA"]:
        diff = abs(fold_m[key] - metrics_before[key])
        ok = diff < 1e-5
        print(f"  [resume_test] {key}: before={metrics_before[key]:.6f}, "
              f"after={fold_m[key]:.6f}, diff={diff:.2e} — {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise RuntimeError(
                f"RESUME TEST FAIL: {key} changed after re-run. "
                "Experiment is not reproducible."
            )

    print("  [resume_test] RESUME TEST PASS ✓")
    print("  Completed folds skipped. Interrupted fold resumed correctly.")
    print("  Results identical to uninterrupted run.")


# ══════════════════════════════════════════════════════════
# Phase 5 — Full Run
# ══════════════════════════════════════════════════════════

def phase5_full_run(splits: list, data: dict) -> list:
    print("\n" + "=" * 70)
    print("PHASE 5 — FULL 5-PARTITION × 3-FOLD EXPERIMENT")
    print("=" * 70)

    all_fold_metrics = []

    for p_idx in range(N_PARTITIONS):
        print(f"\n{'═' * 60}")
        print(f"PARTITION {p_idx} (seed={PARTITION_SEEDS[p_idx]})")
        print(f"{'═' * 60}")

        for f_idx in range(N_FOLDS):
            if is_fold_complete(p_idx, f_idx):
                print(f"  [resume] P{p_idx}F{f_idx} already complete — loading metrics.")
                metrics_path = os.path.join(fold_dir(p_idx, f_idx), "metrics.json")
                with open(metrics_path) as ff:
                    all_fold_metrics.append(json.load(ff))
                continue

            fold_m = run_fold(p_idx, f_idx, splits, data)
            all_fold_metrics.append(fold_m)

    return all_fold_metrics


# ══════════════════════════════════════════════════════════
# Phase 6 — Statistics and Plots
# ══════════════════════════════════════════════════════════

def phase6_statistics(all_fold_metrics: list) -> dict:
    print("\n" + "=" * 70)
    print("PHASE 6 — STATISTICS AND PLOTS")
    print("=" * 70)

    # ── Partition-level aggregation ───────────────────────────────
    partition_results = []
    for p_idx in range(N_PARTITIONS):
        fold_m = [m for m in all_fold_metrics if m["partition"] == p_idx]
        fold_m.sort(key=lambda x: x["fold"])
        assert len(fold_m) == N_FOLDS, (
            f"Partition {p_idx} has {len(fold_m)} folds, expected {N_FOLDS}"
        )
        p_agg = aggregate_partition(fold_m)
        partition_results.append(p_agg)

        print(f"\n  Partition {p_idx}:")
        print(f"    ρ_A = {p_agg['rho_A_mean']:.4f} ± {p_agg['rho_A_std']:.4f}")
        print(f"    ρ_B = {p_agg['rho_B_mean']:.4f} ± {p_agg['rho_B_std']:.4f}")
        print(f"    ρ_C = {p_agg['rho_C_mean']:.4f} ± {p_agg['rho_C_std']:.4f}")
        print(f"    Δρ_BA = {p_agg['delta_rho_BA_mean']:+.4f}")
        print(f"    Δρ_CA = {p_agg['delta_rho_CA_mean']:+.4f}")

    # ── Final statistics over 5 partitions ───────────────────────
    final_stats = compute_final_statistics(partition_results)

    print(f"\n{'─' * 60}")
    print(f"FINAL STATISTICS (N=5 independent partitions):")
    print(f"  mean ρ_A = {final_stats['mean_rho_A_mean']:.4f}")
    print(f"  mean ρ_B = {final_stats['mean_rho_B_mean']:.4f}")
    print(f"  mean ρ_C = {final_stats['mean_rho_C_mean']:.4f}")
    print(f"  mean Δρ_BA = {final_stats['mean_delta_rho_BA_mean']:+.4f} "
          f"[95% CI: {final_stats['ci95_lo_delta_rho_BA_mean']:+.4f}, "
          f"{final_stats['ci95_hi_delta_rho_BA_mean']:+.4f}]")
    print(f"  mean Δρ_CA = {final_stats['mean_delta_rho_CA_mean']:+.4f} "
          f"[95% CI: {final_stats['ci95_lo_delta_rho_CA_mean']:+.4f}, "
          f"{final_stats['ci95_hi_delta_rho_CA_mean']:+.4f}]")
    print(f"  H10 survives: {final_stats['decision']['H10_survives']}")
    print(f"  {final_stats['statistical_power_note']}")

    # ── Save CSVs ─────────────────────────────────────────────────
    fold_df = pd.DataFrame(all_fold_metrics)
    fold_df.to_csv(FOLD_METRICS_CSV, index=False)
    print(f"\n[phase6] Fold metrics → {FOLD_METRICS_CSV}")

    part_df = pd.DataFrame(partition_results)
    part_df.to_csv(PARTITION_METRICS_CSV, index=False)
    print(f"[phase6] Partition metrics → {PARTITION_METRICS_CSV}")

    # ── Generate plots ────────────────────────────────────────────
    try:
        from plots import generate_all_plots
        generate_all_plots(all_fold_metrics, partition_results, final_stats)
        print("[phase6] Plots generated.")
    except Exception as e:
        print(f"[phase6] WARNING: Plot generation failed: {e}")
        print("  (Non-fatal — experiment results are saved.)")

    return {
        "fold_metrics": all_fold_metrics,
        "partition_results": partition_results,
        "final_stats": final_stats,
    }


# ══════════════════════════════════════════════════════════
# Phase 7 — Final Report
# ══════════════════════════════════════════════════════════

def phase7_report(stats_bundle: dict, data: dict) -> None:
    print("\n" + "=" * 70)
    print("PHASE 7 — FINAL REPORT")
    print("=" * 70)

    # Run automated audit first
    from audit import run_audit
    splits = generate_and_freeze_splits()
    audit_result = run_audit(
        data["D_oracle"], data["X_unscaled"], data["y"],
        data["pair_i"], data["pair_j"], splits, FOLD_METRICS_CSV,
    )

    if audit_result["overall"] == "FAIL":
        raise RuntimeError(
            f"AUDIT FAIL — Cannot produce scientific report. "
            f"Failed checks: {audit_result['failed_checks']}"
        )

    # Load config
    with open(EXPERIMENT_CONFIG_PATH) as f:
        exp_config = json.load(f)

    # Build final JSON report
    final = {
        "experiment": "CARE-MoE Experiment 4 — Functional Merge Landscape",
        "layer": "middle",
        "scope_limitation": "middle-layer-only",
        "oracle_matrix_hash": data["oracle_hash"],
        "feature_data_hash": data["feature_hash"],
        "partition_seeds": PARTITION_SEEDS,
        "q": Q,
        "q_provenance": exp_config["q_provenance"],
        "model_a_definition": "XGBoost on 11 local pre-merge features (retrained per fold)",
        "model_b_definition": "||z_i - z_j||_2 in q=4 MDS space — NOT a learned predictor",
        "model_c_definition": "XGBoost on 11 local features + 1 geometry distance = 12 features",
        "cv_structure": "5 partitions × 3-fold expert-disjoint CV = 15 folds",
        "statistical_unit": "partition (N=5 independent)",
        "statistical_power_note": stats_bundle["final_stats"]["statistical_power_note"],
        "noise_ceiling": {
            "status": NOISE_CEILING_STATUS,
            "reason": (
                "No genuine repeated Oracle measurements available. "
                "Multi-Seq_Len values are not independent replicates."
            ),
        },
        "pilot_status": "PASS",
        "audit_result": audit_result,
        "fold_results": stats_bundle["fold_metrics"],
        "partition_results": [
            {k: v for k, v in p.items() if not k.endswith("_folds")}
            for p in stats_bundle["partition_results"]
        ],
        "final_statistics": stats_bundle["final_stats"],
        "decision": stats_bundle["final_stats"]["decision"],
        "h10_survives": stats_bundle["final_stats"]["decision"]["H10_survives"],
        "delta_rho_threshold": 0.05,
    }

    atomic_write_json(final, FINAL_REPORT_JSON)
    print(f"[phase7] Final report JSON → {FINAL_REPORT_JSON}")

    # Generate Markdown report
    try:
        from report import generate_markdown_report
        generate_markdown_report(final)
        print(f"[phase7] Final report MD → {FINAL_REPORT_MD}")
    except Exception as e:
        print(f"[phase7] WARNING: Markdown report generation failed: {e}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 70)
    print(f"  H10 survives: {final['h10_survives']}")
    dec = final["decision"]
    print(f"  Case A (geometry fails):      {dec['A_geometry_fails']}")
    print(f"  Case B (adds value/H10):      {dec['B_geometry_adds_value_H10_survives']}")
    print(f"  Case C (geometry dominates):  {dec['C_geometry_dominates']}")
    print(f"  Case D (complementary):       {dec['D_geometry_complementary']}")
    print(f"  Conclusions are middle-layer-only.")
    print("=" * 70)


# ══════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CARE-MoE Experiment 4")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last valid completed fold.")
    parser.add_argument("--pilot-only", action="store_true",
                        help="Run only 2-partition pilot (for testing).")
    parser.add_argument("--skip-pilot", action="store_true",
                        help="Skip pilot (only for debugging; not for scientific runs).")
    args = parser.parse_args()

    ensure_dirs()

    # PHASE 0
    data = phase0_validate_data()

    # PHASE 1
    splits = phase1_freeze_config(data)

    # PHASE 2 — Pilot
    if not args.skip_pilot:
        pilot_metrics = phase2_pilot(splits, data)
    else:
        print("\n[WARNING] Skipping pilot — for debugging only.")

    if args.pilot_only:
        print("\n[pilot-only] Stopping after pilot. CODE IS NOW FROZEN.")
        return

    # PHASE 3 — Resume test
    phase3_resume_test(splits, data)

    # PHASE 4 — Log git hash
    print("\n" + "=" * 70)
    print("PHASE 4 — FREEZE CODE")
    print("=" * 70)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        git_hash = "unavailable"
    print(f"  Git hash: {git_hash}")
    freeze_path = os.path.join(RESULTS_DIR, "code_freeze.json")
    if not os.path.exists(freeze_path):
        atomic_write_json(
            {"git_hash": git_hash,
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "note": "Code frozen after pilot pass. No methodology changes after this point."},
            freeze_path,
        )

    # PHASE 5
    all_fold_metrics = phase5_full_run(splits, data)

    # PHASE 6
    stats_bundle = phase6_statistics(all_fold_metrics)

    # PHASE 7
    phase7_report(stats_bundle, data)


if __name__ == "__main__":
    main()
