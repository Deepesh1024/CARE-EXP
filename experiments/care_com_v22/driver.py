"""
CARE-COM v2.2 Driver — orchestrates all baselines via subprocess isolation.

Each baseline runs in its own Python process so that CUDA memory is fully
reclaimed by the OS between runs. This eliminates VRAM fragmentation OOM
errors on 24GB GPUs.

Supports resumption: if a trajectory JSON already exists, that step is skipped.

Usage:
    PYTHONPATH=. python -m experiments.care_com_v22.driver
"""
import os
import subprocess
import sys

from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.analysis import generate_analysis


def run_isolated(method, seed=None):
    """Runs a single baseline in a fresh subprocess."""
    cmd = [
        sys.executable, "-m", "experiments.care_com_v22.worker",
        "--method", method,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"\n{'='*60}")
    print(f"  Launching isolated process: {method}" + (f" seed={seed}" if seed else ""))
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n[ERROR] Worker {method} (seed={seed}) exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n[Driver] {method} (seed={seed}) completed successfully.")


def checkpoint_exists(config, method, seed=None):
    """Check if a trajectory file already exists for this method."""
    if method == "random":
        path = os.path.join(config.trajectories_dir, f"random_seed_{seed}.json")
    elif method == "static":
        path = os.path.join(config.trajectories_dir, "static_v1.json")
    elif method == "adaptive":
        path = os.path.join(config.trajectories_dir, "adaptive_v21.json")
    else:
        return False

    exists = os.path.exists(path)
    if exists:
        print(f"[Checkpoint] {method} (seed={seed}) already completed: {path} — skipping.")
    return exists


def run_care_com_v22_pilot():
    config = CareComV22Config()

    # 1. Random Baselines (3 seeds, each in its own process)
    for seed in config.random_seeds:
        if not checkpoint_exists(config, "random", seed):
            run_isolated("random", seed=seed)

    # 2. Static v1 CARE-COM Baseline
    if not checkpoint_exists(config, "static"):
        run_isolated("static")

    # 3. Adaptive v2.1 CARE-COM Baseline
    if not checkpoint_exists(config, "adaptive"):
        run_isolated("adaptive")

    # 4. Analysis
    print("\n[v2.2] All baselines completed. Generating analysis tables and figures...")
    generate_analysis(config)


if __name__ == "__main__":
    run_care_com_v22_pilot()
