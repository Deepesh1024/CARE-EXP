"""
EXPERIMENT 3C — ORCHESTRATOR
==============================
Runs the complete 4-checkpoint longitudinal data generation pipeline:

  Phase 0: Pair Manifest Generation (from 3B Oracle matrices)
  Phase 1: Calibration Corpus Generation (Salesforce/wikitext)
  Phase 2: Oracle Distance Matrix Generation (priority-ordered)
  Phase 3: Final Dataset Validation
"""

import os
import sys
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phase(script_name, description):
    print("\n" + "=" * 70)
    print(f"STARTING: {description}")
    print("=" * 70)

    script_path = os.path.join(_THIS_DIR, script_name)
    result = subprocess.run([sys.executable, script_path], cwd=_THIS_DIR)

    if result.returncode != 0:
        print(f"\n[FATAL] {script_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    # Archive old 8-checkpoint results if they exist
    old_checkpoints = os.path.join(_THIS_DIR, "checkpoints")
    archive_dir = os.path.join(_THIS_DIR, "checkpoints_archive_8ckpt")
    if os.path.exists(old_checkpoints) and not os.path.exists(archive_dir):
        print(f"[Archive] Moving old 8-checkpoint results to {archive_dir}")
        os.rename(old_checkpoints, archive_dir)

    print("=" * 70)
    print("CARE-MoE EXPERIMENT 3C: LONGITUDINAL DATA GENERATION")
    print("  Design: 4 checkpoints (10%, 40%, 70%, 100%)")
    print("  Priority: 100% first (complete), then 10/40/70 (sampled)")
    print("=" * 70)

    run_phase("phase0_pair_manifest.py", "Phase 0: Pair Manifest Generation")
    run_phase("phase1_calibration.py", "Phase 1: Calibration Corpus Generation")
    run_phase("phase2_data_generation.py", "Phase 2: Oracle Distance Matrix Generation")
    run_phase("phase3_validation.py", "Phase 3: Final Dataset Validation")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3C DATA GENERATION PIPELINE COMPLETED")
    print("=" * 70)
    print("If validation PASSED, the dataset is frozen and ready for analysis.")
    print("Check results/final_validation_report.txt for details.")


if __name__ == "__main__":
    main()
