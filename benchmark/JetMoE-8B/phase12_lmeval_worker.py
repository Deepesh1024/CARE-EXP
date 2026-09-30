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

# HumanEval requires this flag -- set before any lm_eval imports
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# ── DIAGNOSTIC ────────────────────────────────────────────────────────────────
print(f"[lmeval_worker] Python:              {sys.executable}", flush=True)
print(f"[lmeval_worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}", flush=True)

# ── STEP 1: Pre-init CUDA ─────────────────────────────────────────────────────
import torch

print(f"[lmeval_worker] torch version:       {torch.__version__}", flush=True)
print(f"[lmeval_worker] torch.version.cuda:  {torch.version.cuda}", flush=True)
print(f"[lmeval_worker] cuda.is_available(): {torch.cuda.is_available()}", flush=True)

if not torch.cuda.is_available():
    try:
        import subprocess as _sp
        _nv = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                      capture_output=True, text=True)
        print(f"[lmeval_worker] nvidia-smi output: {_nv.stdout.strip() or _nv.stderr.strip()}", flush=True)
    except Exception as _e:
        print(f"[lmeval_worker] nvidia-smi failed: {_e}", flush=True)
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
