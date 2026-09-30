"""
Phase 12 lm-eval Worker
========================
Called as a subprocess by phase12_evaluate_multicapability.py.
Pre-initializes CUDA before lm_eval runs to avoid the 'caching_allocator_warmup'
failure that occurs when lm_eval tries to query GPU memory before the CUDA context
is created in the subprocess.

Usage (called automatically by phase12):
    python phase12_lmeval_worker.py <model_path> <output_dir>
"""

import sys
import os

# ── STEP 1: Pre-init CUDA ─────────────────────────────────────────────────────
# caching_allocator_warmup in newer transformers calls torch.cuda.mem_get_info()
# before the CUDA context exists, which crashes with "No CUDA GPUs are available".
# Calling torch.cuda.init() here forces the CUDA context to be created first.
import torch

if not torch.cuda.is_available():
    print("[lmeval_worker] ERROR: torch.cuda.is_available() returned False!", flush=True)
    sys.exit(1)

print(f"[lmeval_worker] CUDA ready: {torch.cuda.get_device_name(0)}", flush=True)
torch.cuda.init()
torch.cuda.empty_cache()
print(f"[lmeval_worker] VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB free", flush=True)

# ── STEP 2: Build lm_eval args and call main ──────────────────────────────────
model_path  = sys.argv[1]
output_dir  = sys.argv[2]
tasks       = sys.argv[3] if len(sys.argv) > 3 else "mmlu,gsm8k,humaneval"

sys.argv = [
    "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
    "--tasks", tasks,
    "--device", "cuda:0",
    "--batch_size", "1",
    "--output_path", output_dir,
    "--trust_remote_code",
    "--confirm_run_unsafe_code",
]

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
