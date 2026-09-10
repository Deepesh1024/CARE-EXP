import os
import json
import torch

# ══════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp7a_51")
EXP7A_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp7a")

DATA_DIR = os.path.join(EXP7A_RESULTS_DIR, "data")
SENSITIVITY_DIR = os.path.join(EXP7A_RESULTS_DIR, "sensitivity")

INTERVENTION_DIR = os.path.join(RESULTS_DIR, "intervention")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Status tracking
TASK_STATUS_FILE = os.path.join(RESULTS_DIR, "task_status.json")

# ══════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════
BASE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
HF_REVISION = "main"

# Model architecture discovered
N_LAYERS = 16
N_EXPERTS = 64
INTERMEDIATE_SIZE = 1024
HIDDEN_SIZE = 2048

# Central layer (floor(16/2))
TARGET_LAYER_IDX = 8

# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════
DATASET_NAME = "allenai/ai2_arc"
DATASET_SUBSET = "ARC-Challenge"
RANDOM_SEED = 7051

SPLIT_SIZES = {
    "D_screen": 1000,
    "D_proxy": 500,
    "D_validation": 500,
    "D_final": 590
}

# ══════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════
BATCH_SIZE = 4
MAX_SEQ_LEN = 1024

# ══════════════════════════════════════════════════════════
# Hardware & Device
# ══════════════════════════════════════════════════════════
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
# Helpers
# ══════════════════════════════════════════════════════════
def ensure_dirs():
    """Create all output directories."""
    for d in [RESULTS_DIR, DATA_DIR, SENSITIVITY_DIR, INTERVENTION_DIR, ANALYSIS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

def load_task_status():
    if os.path.exists(TASK_STATUS_FILE):
        with open(TASK_STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_task_status(status):
    with open(TASK_STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

def mark_task(task_id, state, details=None):
    status = load_task_status()
    import datetime
    status[task_id] = {
        "state": state,
        "details": details or "",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_task_status(status)

def is_task_completed(task_id):
    status = load_task_status()
    return status.get(task_id, {}).get("state") == "completed"
