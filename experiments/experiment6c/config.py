import os
import json
import torch

# ==============================================================
# PROJECT ARCHITECTURE & PATHS
# ==============================================================
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
DIRS = {
    "root": RESULTS_DIR,
    "config": os.path.join(RESULTS_DIR, "config"),
    "capability_basis": os.path.join(RESULTS_DIR, "capability_basis"),
    "raw": os.path.join(RESULTS_DIR, "raw"),
    "processed": os.path.join(RESULTS_DIR, "processed"),
    "expert_vectors": os.path.join(RESULTS_DIR, "expert_vectors"),
    "token_vectors": os.path.join(RESULTS_DIR, "token_vectors"),
    "routing": os.path.join(RESULTS_DIR, "routing"),
    "trajectories": os.path.join(RESULTS_DIR, "trajectories"),
    "pair_metrics": os.path.join(RESULTS_DIR, "pair_metrics"),
    "models": os.path.join(RESULTS_DIR, "models"),
    "nulls": os.path.join(RESULTS_DIR, "nulls"),
    "plots": os.path.join(RESULTS_DIR, "plots"),
    "tables": os.path.join(RESULTS_DIR, "tables"),
    "logs": os.path.join(RESULTS_DIR, "logs"),
}

def ensure_dirs():
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

# ==============================================================
# HARDWARE & MODEL SPECS
# ==============================================================
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
DTYPE = torch.bfloat16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    DEVICE = "mps"

NUM_LAYERS = 16
NUM_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 8

CHECKPOINTS = {
    "checkpoint_10": "step120000-tokens503B",
    "checkpoint_40": "step490000-tokens2055B",
    "checkpoint_70": "step795000-tokens3334B",
    "checkpoint_100": "main"
}

# Ordered transitions for trajectory analysis
TRANSITIONS = [
    ("checkpoint_10", "checkpoint_40"),
    ("checkpoint_40", "checkpoint_70"),
    ("checkpoint_70", "checkpoint_100")
]

# ==============================================================
# 10 EMPIRICAL CAPABILITY / TASK AXES (AUDITED)
# ==============================================================
# Removed ambiguous axes (global_facts, us_foreign_policy).
# astronomy and formal_logic retained with documented structural rationale.
CAPABILITY_AXES = {
    "axis_1_math": {
        "source": "cais/mmlu",
        "categories": ["college_mathematics", "high_school_mathematics", "elementary_mathematics"],
        "max_samples": 300
    },
    "axis_2_physics_astro": {
        "source": "cais/mmlu",
        "categories": ["college_physics", "astronomy"],
        "max_samples": 300
    },
    "axis_3_bio_med": {
        "source": "cais/mmlu",
        "categories": ["anatomy", "college_medicine", "clinical_knowledge"],
        "max_samples": 300
    },
    "axis_4_cs_eng": {
        "source": "cais/mmlu",
        "categories": ["college_computer_science", "machine_learning"],
        "max_samples": 300
    },
    "axis_5_law": {
        "source": "cais/mmlu",
        "categories": ["jurisprudence", "professional_law", "international_law"],
        "max_samples": 300
    },
    "axis_6_history": {
        "source": "cais/mmlu",
        "categories": ["high_school_european_history", "high_school_world_history"],
        "max_samples": 300
    },
    "axis_7_philosophy_logic": {
        "source": "cais/mmlu",
        "categories": ["philosophy", "formal_logic", "moral_disputes"],
        "max_samples": 300
    },
    "axis_8_business_econ": {
        "source": "cais/mmlu",
        "categories": ["econometrics", "high_school_macroeconomics", "business_ethics"],
        "max_samples": 300
    },
    "axis_9_psychology_soc": {
        "source": "cais/mmlu",
        "categories": ["sociology", "professional_psychology", "high_school_psychology"],
        "max_samples": 300
    },
    "axis_10_general_reasoning": {
        "source": "ai2_arc",
        "categories": ["ARC-Challenge"],
        "max_samples": 300
    }
}

def save_capability_basis_spec():
    ensure_dirs()
    spec_path = os.path.join(DIRS["capability_basis"], "EXP6C_CAPABILITY_BASIS.json")
    with open(spec_path, "w") as f:
        json.dump(CAPABILITY_AXES, f, indent=4)

if __name__ == "__main__":
    ensure_dirs()
    save_capability_basis_spec()
    print("[SUCCESS] config.py executed successfully.")
