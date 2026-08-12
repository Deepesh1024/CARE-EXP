"""
CARE-MoE Experiment 4 — Noise Ceiling
=======================================
Spec §13, User correction §1 and §7:

Status: SKIPPED

Reason: No genuine repeated Oracle measurements are available.

The multi-Seq_Len values (64, 128, 256, 512) in output.json are NOT
independent replicates of the same measurement. Different sequence
lengths produce different Oracle KL values because:
  - More tokens = different routing patterns
  - Different calibration distributions
  - Not i.i.d. replicates of the same underlying quantity

Therefore using multi-Seq_Len as a noise proxy would fabricate a
reliability estimate that has no valid statistical interpretation.

This module exists to:
  1. Formally document the SKIPPED status.
  2. Provide a placeholder that will be activated if genuine
     repeated measurements are provided in the future.
  3. Produce a noise_ceiling/status.json file.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NOISE_CEILING_STATUS,
    NOISE_CEILING_REASON,
    NOISE_CEILING_DIR,
)


NOISE_CEILING_RESULT = {
    "status": NOISE_CEILING_STATUS,
    "reason": NOISE_CEILING_REASON,
    "bins": None,
    "global_rho_max": None,
    "per_bin_rho_max": None,
    "methodology": (
        "Noise ceiling requires genuine independent repeated Oracle measurements "
        "of the same (expert_i, expert_j) pair under identical conditions. "
        "Such measurements are not available in this dataset. "
        "Multi-Seq_Len values are calibration variants, not replicates."
    ),
    "how_to_activate": (
        "To activate the noise ceiling: provide a file "
        "repeated_oracle_measurements.json containing at minimum 2 independent "
        "Oracle KL measurements for each of ~100 pairs, stratified across the "
        "empirical Oracle distance distribution. Then set NOISE_CEILING_STATUS "
        "= 'ACTIVE' in config.py and re-run noise_ceiling.py."
    ),
}


def run_noise_ceiling() -> dict:
    """Write noise ceiling status and return result dict."""
    os.makedirs(NOISE_CEILING_DIR, exist_ok=True)
    status_path = os.path.join(NOISE_CEILING_DIR, "status.json")

    with open(status_path, "w") as f:
        json.dump(NOISE_CEILING_RESULT, f, indent=2)

    print(f"[noise_ceiling] Status: {NOISE_CEILING_STATUS}")
    print(f"  Reason: {NOISE_CEILING_REASON}")
    print(f"  Written → {status_path}")

    return NOISE_CEILING_RESULT


if __name__ == "__main__":
    run_noise_ceiling()
