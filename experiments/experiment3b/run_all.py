"""
CARE-MoE Experiment 3B — Run All Phases
=========================================
Orchestrates the complete Capability Geometry Validation Phase A pipeline.

Usage:
    cd experiments/experiment3b
    python run_all.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import traceback


def run_phase(name: str, module_name: str) -> None:
    """Run a single phase and track timing."""
    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"{'#' * 70}\n")

    start = time.time()
    try:
        module = __import__(module_name)
        module.main()
    except Exception as e:
        print(f"\n[ERROR] {name} failed: {e}")
        traceback.print_exc()
        raise
    elapsed = time.time() - start
    print(f"\n[TIMING] {name}: {elapsed:.1f}s ({elapsed/60:.1f}m)")


def main():
    overall_start = time.time()

    print("=" * 70)
    print("CARE-MoE EXPERIMENT 3B — CAPABILITY GEOMETRY VALIDATION PHASE A")
    print("=" * 70)
    print()

    phases = [
        ("Phase 1: Distance Matrix Construction", "phase1_distance_matrix"),
        ("Phase 2: SMACOF MDS + Null Models", "phase2_mds_nulls"),
        ("Phase 3: Expert-Level Cross-Validation", "phase3_cross_validation"),
        ("Phase 4: Statistical Analysis & Figures", "phase4_analysis"),
        ("Phase 5: Final Report & Audit", "phase5_report"),
    ]

    for name, module_name in phases:
        run_phase(name, module_name)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 3B COMPLETE — Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}m)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
