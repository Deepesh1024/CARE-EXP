import os
import torch

# ============================================================
# EXPERIMENT 3C CONFIGURATION — REVISED 4-CHECKPOINT DESIGN
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# ── Directories ────────────────────────────────────
DATA_DIR = os.path.join(_THIS_DIR, "data")
CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Archive directory for the old 8-checkpoint partial results
ARCHIVE_DIR = os.path.join(_THIS_DIR, "checkpoints_archive_8ckpt")

# ── 1. Model & Target Architecture ─────────────────
BASE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_EXPERTS = 64
LAYERS = ["first", "middle", "last"]
LAYER_INDICES = {"first": 0, "middle": 8, "last": 15}
TOTAL_MOE_LAYERS = 16

# ── 2. Checkpoint Definitions (4 strategic stages) ──
# 10% = early training
# 40% = developing structure
# 70% = mature structure (nearest verified: step795000 = ~65.2%)
# 100% = final trained model
CHECKPOINTS = {
    "checkpoint_10": {
        "hf_revision": "step120000-tokens503B",
        "target_pct": 10,
        "actual_step": 120000,
        "actual_tokens_B": 503,
        "actual_pct": 9.8,
        "coverage": "sampled",
    },
    "checkpoint_40": {
        "hf_revision": "step490000-tokens2055B",
        "target_pct": 40,
        "actual_step": 490000,
        "actual_tokens_B": 2055,
        "actual_pct": 40.2,
        "coverage": "sampled",
    },
    "checkpoint_70": {
        "hf_revision": "step795000-tokens3334B",
        "target_pct": 70,
        "actual_step": 795000,
        "actual_tokens_B": 3334,
        "actual_pct": 65.2,
        "coverage": "sampled",
        "note": "Nearest verified checkpoint to 70%. Actual = 65.2%.",
    },
    "checkpoint_100": {
        "hf_revision": "main",
        "target_pct": 100,
        "actual_step": 1220000,
        "actual_tokens_B": 5138,
        "actual_pct": 100.0,
        "coverage": "full",
    },
}

# Execution priority order (100% first, then chronological)
CHECKPOINT_PRIORITY = [
    "checkpoint_100",
    "checkpoint_10",
    "checkpoint_40",
    "checkpoint_70",
]

# ── 3. Pair Coverage ───────────────────────────────
FULL_PAIRS = 2016          # C(64,2) for 100% checkpoint
SAMPLED_PAIRS = 384        # Per layer for early checkpoints (96 per quartile)

# ── 4. Calibration Corpus ──────────────────────────
# MUST match Experiment 1/3B methodology exactly
CALIBRATION_DATASET = "Salesforce/wikitext"
CALIBRATION_SUBSET = "wikitext-2-raw-v1"
CALIBRATION_SPLIT = "train"
CALIBRATION_SEQ_LEN = 512
CALIBRATION_N_SEQUENCES = 98
TARGET_CALIBRATION_TOKENS = CALIBRATION_N_SEQUENCES * CALIBRATION_SEQ_LEN  # 50,176
CALIBRATION_CACHE_FILE = os.path.join(CALIBRATION_DIR, "calibration_3c_wikitext.pt")
CALIBRATION_METADATA_FILE = os.path.join(CALIBRATION_DIR, "calibration_3c_wikitext_meta.json")

# ── 5. Execution Parameters ────────────────────────
RANDOM_SEED = 42
GPU_ID = int(os.environ.get("CARE_MOE_GPU_ID", 0))
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CALIB_BATCH_SIZE = 4

# ── 6. 3B Reference Data (for pair manifest) ──────
EXP3B_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp3b")
EXP3B_MATRIX_FILES = {
    "first": os.path.join(EXP3B_RESULTS_DIR, "oracle_distance_matrix_first.csv"),
    "middle": os.path.join(EXP3B_RESULTS_DIR, "oracle_distance_matrix_middle.csv"),
    "last": os.path.join(EXP3B_RESULTS_DIR, "oracle_distance_matrix_last.csv"),
}

# ── 7. Output Paths ───────────────────────────────
PAIR_MANIFEST_FILE = os.path.join(RESULTS_DIR, "3c_pair_manifest.json")
CHECKPOINT_METADATA_FILE = os.path.join(RESULTS_DIR, "checkpoint_metadata.json")
RUNTIME_REPORT_FILE = os.path.join(RESULTS_DIR, "runtime_report.json")
VALIDATION_REPORT_FILE = os.path.join(RESULTS_DIR, "final_validation_report.txt")

# ── 8. Oracle Definition ──────────────────────────
# D(E_i, E_j) = KL( P_orig || P_merged )
# P_merged = model output after replacing E_i and E_j with UniformAverage(E_i, E_j)
# Identical to validated Experiment 3B implementation in CARE_MoE_V3_E1.py

def ensure_dirs():
    for d in [CALIBRATION_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    for ckpt_name in CHECKPOINTS:
        for layer in LAYERS:
            os.makedirs(os.path.join(RESULTS_DIR, ckpt_name, layer), exist_ok=True)
