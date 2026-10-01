"""
Fix PyTorch CUDA Version
========================
Reinstalls PyTorch compiled for CUDA 12.1 to match the system CUDA runtime.

Run from the root of Experiments-V3 on the VM:
    python benchmark/JetMoE-8B/fix_torch_cuda.py
"""

import subprocess
import sys
import os

DIVIDER = "=" * 60

def run(cmd, **kwargs):
    print(f"\n  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)

print(f"\n{DIVIDER}")
print("  TORCH CUDA VERSION FIX (cu121)")
print(DIVIDER)

# 1. Show current state
print("\n[1/4] Current PyTorch install:")
run([sys.executable, "-c",
     "import torch; print('torch:', torch.__version__); "
     "print('cuda built for:', torch.version.cuda); "
     "print('cuda available:', torch.cuda.is_available())"])

# 2. Uninstall current torch
print("\n[2/4] Uninstalling current torch...")
run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])

# 3. Install torch with CUDA 12.1
print("\n[3/4] Installing torch+cu121...")
run([
    sys.executable, "-m", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

# 4. Verify
print("\n[4/4] Verifying new install...")
result = subprocess.run(
    [sys.executable, "-c",
     "import torch; "
     "print('torch version:', torch.__version__); "
     "print('cuda built for:', torch.version.cuda); "
     "print('cuda available:', torch.cuda.is_available()); "
     "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])

if "cuda available: True" in result.stdout:
    print(f"\n{DIVIDER}")
    print("  SUCCESS! CUDA is now working.")
    print("  Run: python benchmark/JetMoE-8B/phase12_evaluate_multicapability.py")
    print(DIVIDER)
else:
    print(f"\n{DIVIDER}")
    print("  WARNING: CUDA still not available after reinstall.")
    print("  Run: nvcc --version  to confirm your CUDA runtime version.")
    print(DIVIDER)
