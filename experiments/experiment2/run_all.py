"""
CARE-MoE Experiment 2 — Master Orchestrator
==============================================
Executes all Experiment 2 phases sequentially:
  - Phase 0: Oracle Feature Audit
  - Phase 0.5: Residual Analysis
  - Phase 0.75: Existing Feature Correlation
  - Phase 1: Capability Descriptor Engineering
  - Phase 2: Descriptor Diagnostics
  - Phase 3: Combined CARE Model Regression
  - Phase 4: Interpretability & Importance
  - Phase 5: Leave-One-Out Ablation
  - Phase 6: Linearization Gap Comparison
"""

import time
import os
import sys

# Ensure this directory is in path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils import set_global_seed, ensure_dirs
import phase0_audit
import phase05_residuals
import phase075_correlation
import phase1_descriptors
import phase2_diagnostics
import phase3_regression
import phase4_interpretability
import phase5_ablation
import phase6_gap


def main():
    start_time = time.time()
    set_global_seed()
    ensure_dirs()

    print("*" * 80)
    print("CARE-MoE Experiment 2: Capability-Aware Descriptor Engineering — Master Pipeline")
    print("*" * 80)

    print("\n>>> Executing Phase 0: Oracle Feature Audit...")
    t0 = time.time()
    phase0_audit.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 0.5: Residual Analysis...")
    t0 = time.time()
    phase05_residuals.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 0.75: Existing Feature Correlation...")
    t0 = time.time()
    phase075_correlation.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 1: Capability Descriptor Engineering...")
    t0 = time.time()
    phase1_descriptors.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 2: Descriptor Diagnostics...")
    t0 = time.time()
    phase2_diagnostics.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 3: Combined CARE Model...")
    t0 = time.time()
    phase3_regression.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 4: Interpretability...")
    t0 = time.time()
    phase4_interpretability.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 5: Leave-One-Out Ablation...")
    t0 = time.time()
    phase5_ablation.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    print("\n>>> Executing Phase 6: Linearization Gap Comparison...")
    t0 = time.time()
    phase6_gap.main()
    print(f"[Completed in {time.time() - t0:.2f}s]")

    total_time = time.time() - start_time
    print("\n" + "*" * 80)
    print(f"Experiment 2 Master Pipeline Successfully Executed in {total_time:.2f} seconds!")
    print("*" * 80)


if __name__ == "__main__":
    main()
