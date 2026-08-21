"""
EXPERIMENT 6B — CENTRAL CONFIGURATION
============================================================
Token / Routing Environment → Functional Evolution of MoE Experts.

All constants, paths, parameters, and resource scheduling for
the two-timescale (coarse checkpoint + fine routing-window)
functional-evolution experiment.
"""

import os
import json
import torch
import hashlib

# ══════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp6b")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TELEMETRY_DIR = os.path.join(RESULTS_DIR, "telemetry")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
REPORTS_DIR = RESULTS_DIR  # Reports live at top level of results

# Existing upstream data
EXP3C_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp3c")
EXP3B_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp3b")
EXP4_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp4")
EXP6A_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp6")

# Calibration corpus (reuse Exp 3C calibration exactly)
EXP3C_CALIB_DIR = os.path.join(_PROJECT_ROOT, "experiments", "experiment3c", "data", "calibration")
CALIBRATION_CACHE_FILE = os.path.join(EXP3C_CALIB_DIR, "calibration_3c_wikitext.pt")

# Status tracking
TASK_STATUS_FILE = os.path.join(RESULTS_DIR, "task_status.json")

# ══════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════
BASE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_EXPERTS = 64
TOTAL_MOE_LAYERS = 16
LAYERS = ["first", "middle", "last"]
LAYER_INDICES = {"first": 0, "middle": 8, "last": 15}
NUM_EXPERTS_PER_TOK = 8  # Top-k

# ══════════════════════════════════════════════════════════
# Checkpoints (from Exp 3C, verified)
# ══════════════════════════════════════════════════════════
CHECKPOINTS = {
    "checkpoint_10": {
        "hf_revision": "step120000-tokens503B",
        "target_pct": 10,
        "actual_step": 120000,
        "actual_tokens_B": 503,
        "actual_pct": 9.8,
    },
    "checkpoint_40": {
        "hf_revision": "step490000-tokens2055B",
        "target_pct": 40,
        "actual_step": 490000,
        "actual_tokens_B": 2055,
        "actual_pct": 40.2,
    },
    "checkpoint_70": {
        "hf_revision": "step795000-tokens3334B",
        "target_pct": 70,
        "actual_step": 795000,
        "actual_tokens_B": 3334,
        "actual_pct": 65.2,
    },
    "checkpoint_100": {
        "hf_revision": "main",
        "target_pct": 100,
        "actual_step": 1220000,
        "actual_tokens_B": 5138,
        "actual_pct": 100.0,
    },
}

CHECKPOINT_ORDER = ["checkpoint_10", "checkpoint_40", "checkpoint_70", "checkpoint_100"]

# ══════════════════════════════════════════════════════════
# q-values from Exp 3B ranking
# ══════════════════════════════════════════════════════════
# Exp 3B dimension_summary.csv — Oracle_rho rankings:
#   middle: q=4 (0.7233), q=6 (0.7137), q=3 (0.7103)
#   last:   q=9 (0.7386), q=8 (0.7301), q=6 (0.7203)
#   first:  q=3 (0.6301), q=4 (0.6257), q=2 (0.6042)
# Primary: q=4 (selected in Exp 3B, fixed in Exp 4)
# Secondary: q=6 (second-best across middle and last)
# Tertiary: q=3 (second-best for first, third-best for middle)
Q_PRIMARY = 4
Q_SECONDARY = 6
Q_TERTIARY = 3
Q_VALUES = [Q_PRIMARY, Q_SECONDARY, Q_TERTIARY]

# ══════════════════════════════════════════════════════════
# MDS / Alignment Parameters
# ══════════════════════════════════════════════════════════
SMACOF_MAX_ITER = 3000
SMACOF_N_INIT = 5
SMACOF_EPS = 1e-4

# ══════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════
CALIBRATION_SEQ_LEN = 512
CALIBRATION_N_SEQUENCES = 98
CALIBRATION_SHA256 = "c7b221ffbd2d00340ae3795639176369eb7c918ef8cf7fbc9cd1416812c139c1"

# ══════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════
RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════
# Resource Scheduling (configurable for target VM)
# ══════════════════════════════════════════════════════════
NUM_GPU_WORKERS = 1
NUM_CPU_WORKERS = 4
MAX_CONCURRENT_CHECKPOINTS = 1
MAX_CONCURRENT_LAYERS = 1
BATCH_SIZE = 4  # Calibration batch size for forward passes
GPU_ID = int(os.environ.get("CARE_MOE_GPU_ID", 0))

def get_device():
    if torch.cuda.is_available():
        return f"cuda:{GPU_ID}"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ══════════════════════════════════════════════════════════
# Fine-Window Configuration
# ══════════════════════════════════════════════════════════
# Window sizes in number of calibration sequences processed.
# Small / Medium / Large — not selected for best result.
WINDOW_SIZES = [10, 25, 50]  # sequences per window

# ══════════════════════════════════════════════════════════
# Tracked pairs (from Exp 3C)
# ══════════════════════════════════════════════════════════
SAMPLED_PAIRS = 384  # Consistently tracked across checkpoints

# ══════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════
def ensure_dirs():
    """Create all output directories."""
    for d in [RESULTS_DIR, PLOTS_DIR, TELEMETRY_DIR, EMBEDDINGS_DIR,
              MODELS_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)
    for ckpt_name in CHECKPOINTS:
        for layer in LAYERS:
            os.makedirs(os.path.join(TELEMETRY_DIR, ckpt_name, layer), exist_ok=True)
    for q in Q_VALUES:
        os.makedirs(os.path.join(EMBEDDINGS_DIR, f"q{q}"), exist_ok=True)


def load_task_status():
    """Load or initialize task status manifest."""
    if os.path.exists(TASK_STATUS_FILE):
        with open(TASK_STATUS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_task_status(status):
    """Save task status manifest."""
    with open(TASK_STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def mark_task(task_id, state, details=None):
    """Mark a task as pending/running/completed/failed."""
    status = load_task_status()
    status[task_id] = {
        "state": state,
        "details": details or "",
        "timestamp": str(torch.cuda.Event) if False else "cpu_time",
    }
    import datetime
    status[task_id]["timestamp"] = datetime.datetime.now().isoformat()
    save_task_status(status)


def is_task_completed(task_id):
    """Check if a task has already completed."""
    status = load_task_status()
    return status.get(task_id, {}).get("state") == "completed"


def get_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
