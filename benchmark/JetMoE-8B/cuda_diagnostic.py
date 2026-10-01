"""
CUDA Diagnostic Script
======================
Run this on the VM to diagnose why lm_eval subprocess fails.
Runs the diagnostic in multiple ways to pinpoint the exact cause.

Usage:
    python benchmark/JetMoE-8B/cuda_diagnostic.py
"""

import subprocess
import sys
import os

CKPT_PATH   = "benchmark_results/JetMoE-8B/checkpoints/jetmoe_base"
WORKER      = "benchmark/JetMoE-8B/phase12_lmeval_worker.py"
OUT_DIR     = "/tmp/test_lmeval_out"
DIVIDER     = "=" * 60

# ──────────────────────────────────────────────────────────────────────────────
# 1. Print current process info
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  STEP 1: THIS PROCESS")
print(DIVIDER)
print(f"  Python:              {sys.executable}")
print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

try:
    import torch
    print(f"  torch version:       {torch.__version__}")
    print(f"  torch.version.cuda:  {torch.version.cuda}")
    print(f"  cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  torch import failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Run the worker directly as a subprocess (no env manipulation)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  STEP 2: DIRECT SUBPROCESS (same as phase12)")
print(DIVIDER)
cmd = [sys.executable, WORKER, CKPT_PATH, OUT_DIR, "mmlu"]
print(f"  Running: {' '.join(cmd)}\n")
result = subprocess.run(cmd, capture_output=False)
print(f"\n  Exit code: {result.returncode}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Run with explicit env copy (the old broken approach)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  STEP 3: SUBPROCESS WITH env=os.environ.copy()")
print(DIVIDER)
env_copy = os.environ.copy()
cmd2 = [sys.executable, WORKER, CKPT_PATH, OUT_DIR + "_2", "mmlu"]
print(f"  Running: {' '.join(cmd2)}\n")
result2 = subprocess.run(cmd2, env=env_copy, capture_output=False)
print(f"\n  Exit code: {result2.returncode}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Inline torch CUDA check (no subprocess)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  STEP 4: INLINE MINI CUDA CHECK")
print(DIVIDER)
mini = [sys.executable, "-c",
        "import torch; print('available:', torch.cuda.is_available()); "
        "print('device count:', torch.cuda.device_count()); "
        "[print(f'GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"]
subprocess.run(mini)

# ──────────────────────────────────────────────────────────────────────────────
# 5. nvidia-smi
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  STEP 5: nvidia-smi")
print(DIVIDER)
subprocess.run(["nvidia-smi"])

print(f"\n{DIVIDER}")
print("  DIAGNOSTIC COMPLETE")
print("  Share the full output above to fix the lm_eval CUDA issue.")
print(DIVIDER + "\n")
