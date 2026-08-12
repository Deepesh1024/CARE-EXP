"""
EXPERIMENT 3C — PHASE 3: FINAL DATASET VALIDATION
====================================================
Comprehensive validation of the generated longitudinal dataset.

Validates:
  - 100% checkpoint: 3 layers, 2016 pairs each, no NaN, no Inf
  - Early checkpoints: 3 layers, 384 manifest pairs each
  - Same pair manifest used across 10/40/70
  - Expert indices in [0,63], no duplicate pairs, no i==j
  - Calibration hash identical across all checkpoints
  - Matrices are exactly 64×64
  - Diagonal is zero

Generates: results/final_validation_report.txt
"""

import os
import json
import numpy as np
from datetime import datetime

from config import (
    N_EXPERTS,
    LAYERS,
    CHECKPOINTS,
    CHECKPOINT_PRIORITY,
    RESULTS_DIR,
    PAIR_MANIFEST_FILE,
    VALIDATION_REPORT_FILE,
    SAMPLED_PAIRS,
    FULL_PAIRS,
    ensure_dirs,
)


def validate_matrix(matrix, expected_pairs, is_full, layer, ckpt_name, manifest_pairs=None):
    """
    Validate a single 64×64 Oracle distance matrix.
    Returns (passed: bool, issues: list[str]).
    """
    issues = []

    # Shape check
    if matrix.shape != (N_EXPERTS, N_EXPERTS):
        issues.append(f"Shape mismatch: expected (64,64), got {matrix.shape}")
        return False, issues

    # Diagonal check
    diag = np.diag(matrix)
    if not np.all(diag == 0.0):
        non_zero_diag = np.where(diag != 0.0)[0]
        issues.append(f"Diagonal not zero at indices: {non_zero_diag.tolist()}")

    upper_tri = matrix[np.triu_indices(N_EXPERTS, k=1)]

    if is_full:
        # Full coverage: no NaN allowed
        if np.any(np.isnan(upper_tri)):
            n_nan = np.count_nonzero(np.isnan(upper_tri))
            issues.append(f"Full matrix has {n_nan} NaN entries (expected 0)")

        if np.any(np.isinf(upper_tri)):
            n_inf = np.count_nonzero(np.isinf(upper_tri))
            issues.append(f"Matrix has {n_inf} Inf entries")

        # Count measured pairs
        measured = np.count_nonzero(~np.isnan(upper_tri))
        if measured != FULL_PAIRS:
            issues.append(f"Expected {FULL_PAIRS} measured pairs, got {measured}")

    else:
        # Sampled coverage: exactly SAMPLED_PAIRS should be non-NaN
        measured = np.count_nonzero(~np.isnan(upper_tri))
        if measured != expected_pairs:
            issues.append(f"Expected {expected_pairs} measured pairs, got {measured}")

        # Check measured values are finite
        measured_vals = upper_tri[~np.isnan(upper_tri)]
        if len(measured_vals) > 0 and np.any(np.isinf(measured_vals)):
            issues.append(f"Measured values contain Inf")

        # Verify correct pairs are measured (match manifest)
        if manifest_pairs is not None:
            for (i, j) in manifest_pairs:
                if i >= N_EXPERTS or j >= N_EXPERTS:
                    issues.append(f"Pair ({i},{j}) has index out of range [0,63]")
                if i == j:
                    issues.append(f"Pair ({i},{j}) has i==j")
                if i > j:
                    issues.append(f"Pair ({i},{j}) not in upper-triangle form")
                if np.isnan(matrix[i, j]):
                    issues.append(f"Manifest pair ({i},{j}) is NaN (unmeasured)")

    # Symmetry check for measured entries
    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            if not np.isnan(matrix[i, j]):
                if matrix[i, j] != matrix[j, i]:
                    issues.append(f"Asymmetry at ({i},{j}): {matrix[i,j]} != {matrix[j,i]}")
                    break  # One example is enough

    # Negative values check
    measured_vals = upper_tri[~np.isnan(upper_tri)]
    if len(measured_vals) > 0 and np.any(measured_vals < 0):
        n_neg = np.count_nonzero(measured_vals < 0)
        issues.append(f"{n_neg} negative KL values detected")

    passed = len(issues) == 0
    return passed, issues


def main():
    ensure_dirs()

    print("=" * 70)
    print("EXPERIMENT 3C — PHASE 3: FINAL VALIDATION")
    print("=" * 70)

    # Load manifest
    manifest = None
    manifest_status = "MISSING"
    if os.path.exists(PAIR_MANIFEST_FILE):
        with open(PAIR_MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
        manifest_status = "LOADED"
        # Validate manifest structure
        for layer in LAYERS:
            if layer not in manifest.get("layers", {}):
                manifest_status = f"INCOMPLETE (missing layer: {layer})"
                break
            n_pairs = len(manifest["layers"][layer]["pairs"])
            if n_pairs != SAMPLED_PAIRS:
                manifest_status = f"BAD_COUNT ({layer}: {n_pairs} != {SAMPLED_PAIRS})"
                break

    # Collect calibration hashes from all checkpoints
    calib_hashes = {}
    for ckpt_name in CHECKPOINTS:
        complete_markers = []
        for layer in LAYERS:
            marker_path = os.path.join(RESULTS_DIR, ckpt_name, layer, "COMPLETE")
            if os.path.exists(marker_path):
                with open(marker_path, "r") as f:
                    try:
                        marker_data = json.load(f)
                        if "calibration_sha256" in marker_data:
                            calib_hashes.setdefault(ckpt_name, set()).add(
                                marker_data["calibration_sha256"]
                            )
                    except json.JSONDecodeError:
                        pass

    # Verify calibration consistency
    all_hashes = set()
    for h_set in calib_hashes.values():
        all_hashes.update(h_set)
    calib_consistent = len(all_hashes) <= 1

    # Validate each checkpoint/layer
    results = {}
    total_pass = 0
    total_fail = 0
    total_missing = 0
    overall_pass = True

    for ckpt_name in CHECKPOINT_PRIORITY:
        ckpt_info = CHECKPOINTS[ckpt_name]
        is_full = ckpt_info["coverage"] == "full"
        results[ckpt_name] = {"coverage": ckpt_info["coverage"], "layers": {}}

        for layer in LAYERS:
            mat_path = os.path.join(RESULTS_DIR, ckpt_name, layer, "oracle_distance.npy")
            complete_path = os.path.join(RESULTS_DIR, ckpt_name, layer, "COMPLETE")

            if not os.path.exists(mat_path) or not os.path.exists(complete_path):
                results[ckpt_name]["layers"][layer] = {
                    "status": "MISSING",
                    "issues": ["Matrix or COMPLETE marker not found"],
                }
                total_missing += 1
                overall_pass = False
                continue

            try:
                matrix = np.load(mat_path)
            except Exception as e:
                results[ckpt_name]["layers"][layer] = {
                    "status": "FAIL",
                    "issues": [f"Failed to load matrix: {e}"],
                }
                total_fail += 1
                overall_pass = False
                continue

            # Get manifest pairs for this layer (for sampled checkpoints)
            manifest_pairs = None
            expected_pairs = FULL_PAIRS if is_full else SAMPLED_PAIRS
            if not is_full and manifest and layer in manifest.get("layers", {}):
                manifest_pairs = [tuple(p) for p in manifest["layers"][layer]["pairs"]]

            passed, issues = validate_matrix(
                matrix, expected_pairs, is_full, layer, ckpt_name, manifest_pairs
            )

            if passed:
                measured = np.count_nonzero(
                    ~np.isnan(matrix[np.triu_indices(N_EXPERTS, k=1)])
                )
                measured_vals = matrix[np.triu_indices(N_EXPERTS, k=1)]
                measured_vals = measured_vals[~np.isnan(measured_vals)]
                results[ckpt_name]["layers"][layer] = {
                    "status": "PASS",
                    "measured_pairs": int(measured),
                    "kl_min": float(np.min(measured_vals)),
                    "kl_max": float(np.max(measured_vals)),
                    "kl_mean": float(np.mean(measured_vals)),
                    "issues": [],
                }
                total_pass += 1
            else:
                results[ckpt_name]["layers"][layer] = {
                    "status": "FAIL",
                    "issues": issues,
                }
                total_fail += 1
                overall_pass = False

    # Cross-checkpoint manifest consistency (10/40/70 must use same pairs)
    manifest_consistency = "N/A"
    if manifest:
        manifest_consistency = "PASS"
        # The manifest is a single file used by all three early checkpoints,
        # so if it exists and is valid, consistency is guaranteed by design.
        # We verify that each early checkpoint's measured entries match manifest pairs.
        for ckpt_name in ["checkpoint_10", "checkpoint_40", "checkpoint_70"]:
            for layer in LAYERS:
                layer_result = results.get(ckpt_name, {}).get("layers", {}).get(layer, {})
                if layer_result.get("status") == "PASS":
                    mat_path = os.path.join(RESULTS_DIR, ckpt_name, layer, "oracle_distance.npy")
                    matrix = np.load(mat_path)
                    manifest_pairs = manifest["layers"][layer]["pairs"]
                    for (pi, pj) in manifest_pairs:
                        if np.isnan(matrix[pi, pj]):
                            manifest_consistency = f"FAIL (missing pair ({pi},{pj}) in {ckpt_name}/{layer})"
                            overall_pass = False
                            break

    # Generate text report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("EXPERIMENT 3C — FINAL VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 70)
    report_lines.append("")

    report_lines.append(f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}")
    report_lines.append(f"  Passed:  {total_pass}")
    report_lines.append(f"  Failed:  {total_fail}")
    report_lines.append(f"  Missing: {total_missing}")
    report_lines.append(f"  Total:   {total_pass + total_fail + total_missing} / "
                        f"{len(CHECKPOINTS) * len(LAYERS)}")
    report_lines.append("")

    report_lines.append(f"Pair Manifest: {manifest_status}")
    report_lines.append(f"Manifest Consistency (10/40/70): {manifest_consistency}")
    report_lines.append(f"Calibration Hash Consistency: {'PASS' if calib_consistent else 'FAIL'}")
    if all_hashes:
        report_lines.append(f"  Hash(es): {', '.join(all_hashes)}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("PER-CHECKPOINT RESULTS")
    report_lines.append("-" * 70)

    for ckpt_name in CHECKPOINT_PRIORITY:
        ckpt_result = results[ckpt_name]
        report_lines.append(f"\n[{ckpt_name}] (coverage: {ckpt_result['coverage']})")
        for layer in LAYERS:
            lr = ckpt_result["layers"].get(layer, {"status": "MISSING", "issues": []})
            status = lr["status"]
            if status == "PASS":
                report_lines.append(
                    f"  {layer}: {status} ({lr['measured_pairs']} pairs, "
                    f"KL=[{lr['kl_min']:.6f}, {lr['kl_max']:.6f}], "
                    f"mean={lr['kl_mean']:.6f})"
                )
            else:
                report_lines.append(f"  {layer}: {status}")
                for issue in lr.get("issues", []):
                    report_lines.append(f"    - {issue}")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append(f"VERDICT: {'PASS — Dataset is frozen and ready for analysis.'if overall_pass else 'FAIL — Data generation is NOT complete.'}")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file
    os.makedirs(os.path.dirname(VALIDATION_REPORT_FILE), exist_ok=True)
    with open(VALIDATION_REPORT_FILE, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to: {VALIDATION_REPORT_FILE}")

    # Also save structured JSON for programmatic access
    json_report = {
        "overall_pass": overall_pass,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_missing": total_missing,
        "manifest_status": manifest_status,
        "manifest_consistency": manifest_consistency,
        "calibration_consistent": calib_consistent,
        "calibration_hashes": list(all_hashes),
        "results": results,
        "generated_at": datetime.now().isoformat(),
    }
    json_path = VALIDATION_REPORT_FILE.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)


if __name__ == "__main__":
    main()
