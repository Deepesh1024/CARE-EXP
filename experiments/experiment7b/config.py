"""
EXPERIMENT 7B - CONFIGURATION
=============================
CARE-COM Merge Failure Analysis and Joint Functional Interaction
Central configuration module containing all paths, model metadata,
sampling constants, decision gate thresholds, and state tracking.
"""

import os
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ══════════════════════════════════════════════════════════
# Paths & Project Structure
# ══════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Results & Output Paths
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp7b")
CANDIDATE_DIR = os.path.join(RESULTS_DIR, "candidates")
MERGES_DIR = os.path.join(RESULTS_DIR, "merges")
INTERVENTION_DIR = os.path.join(RESULTS_DIR, "interventions")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Status tracking
TASK_STATUS_FILE = os.path.join(RESULTS_DIR, "task_status.json")

# External Experiment Artifacts (Read-Only)
EXP1_OUTPUT_JSON = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")
EXP4_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp4")
EXP5_MODEL_PATH = os.path.join(_PROJECT_ROOT, "results", "exp5", "care_com_model.json")
EXP5_SCALER_PATH = os.path.join(_PROJECT_ROOT, "results", "exp5", "care_com_scaler.joblib")
EXP7A_DATA_DIR = os.path.join(_PROJECT_ROOT, "results", "exp7a", "data")

# ══════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════
BASE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
HF_REVISION = "main"

N_LAYERS = 16
N_EXPERTS = 64
INTERMEDIATE_SIZE = 1024
HIDDEN_SIZE = 2048
TOP_K_EXPERTS = 8

# Central Layer for CARE-COM evaluation (Layer 8)
TARGET_LAYER_IDX = 8

# ══════════════════════════════════════════════════════════
# Feature Definitions (11 Local Features from Exp 1 & Exp 2)
# ══════════════════════════════════════════════════════════
ORIGINAL_FEATURES = [
    "Weight_Distance",
    "Weight_Cosine",
    "Activation_Similarity",
    "Output_Similarity",
    "Routing_Similarity",
    "Usage_Frequency",
    "Jaccard_Overlap",
]

NEW_DESCRIPTORS = [
    "Usage_Asymmetry",
    "Routing_JSD_Proxy",
    "Routing_NPMI_Proxy",
    "Specialization_Diff",
]

LOCAL_FEATURES = ORIGINAL_FEATURES + NEW_DESCRIPTORS

# ══════════════════════════════════════════════════════════
# Candidate Sampling Parameters
# ══════════════════════════════════════════════════════════
RANDOM_SEED = 42
TOTAL_PAIRS_LAYER8 = 2016  # C(64, 2)

# Stratified Sampling: 3 balanced groups of 6 pairs each = 18 pairs
N_PER_STRATUM = 6
STRATA = {
    "Group_A": {"name": "Safe", "pct_min": 0.0, "pct_max": 0.15},       # Top 15% safest predicted
    "Group_B": {"name": "Moderate", "pct_min": 0.40, "pct_max": 0.60},   # Middle 40%-60%
    "Group_C": {"name": "Unsafe", "pct_min": 0.85, "pct_max": 1.00},     # Bottom 15% most damaging
}

# ══════════════════════════════════════════════════════════
# Evaluation & Dataset Configurations
# ══════════════════════════════════════════════════════════
BATCH_SIZE = int(os.environ.get("CARE_MOE_BATCH_SIZE", 4))
MAX_SEQ_LEN = 512

# Primary In-Domain Dataset
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "train"
EVAL_TOKENS_LIMIT = 262144  # 512 sequences of 512 tokens, matching Exp 1 exactly

# Downstream Capability Dataset (AI2 ARC-Challenge)
ARC_DATASET_NAME = "allenai/ai2_arc"
ARC_DATASET_SUBSET = "ARC-Challenge"

# ══════════════════════════════════════════════════════════
# Decision Gates & Statistical Thresholds
# ══════════════════════════════════════════════════════════
BOOTSTRAP_N_SAMPLES = 10000
BOOTSTRAP_CI_LEVEL = 0.95

# Gate 1: Baseline Predictive Utility of CARE-COM
GATE1_MIN_RHO = 0.30
GATE1_ALPHA = 0.05

# Gate 2: Joint Interaction Association with Prediction Error
GATE2_MIN_RHO = 0.40
GATE2_ALPHA = 0.05

# ══════════════════════════════════════════════════════════
# Hardware & Device Management
# ══════════════════════════════════════════════════════════
GPU_ID = int(os.environ.get("CARE_MOE_GPU_ID", 0))

def get_device():
    env_dev = os.environ.get("CARE_MOE_DEVICE")
    if env_dev:
        return env_dev
    if torch is not None and torch.cuda.is_available():
        return f"cuda:{GPU_ID}"
    elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
if torch is not None:
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
else:
    DTYPE = "bfloat16"

# ══════════════════════════════════════════════════════════
# State Tracking & Directory Helpers
# ══════════════════════════════════════════════════════════
def ensure_dirs():
    """Ensure all required results directories exist."""
    for d in [RESULTS_DIR, CANDIDATE_DIR, MERGES_DIR, INTERVENTION_DIR, ANALYSIS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

def load_task_status():
    if os.path.exists(TASK_STATUS_FILE):
        try:
            with open(TASK_STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
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
