import os
import json
import torch

DIRS = {
    "root": os.path.dirname(os.path.abspath(__file__)),
    "exp6c_root": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "exp6c"),
    "exp6c_expert_vectors": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "exp6c", "expert_vectors"),
    "results": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "exp6d"),
}

def ensure_dirs():
    os.makedirs(DIRS["results"], exist_ok=True)
    os.makedirs(os.path.join(DIRS["results"], "plots"), exist_ok=True)

# Hardware & Model
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REVISION = "main"
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 # fp32 causes OOM on 24GB GPUs for 7B models
NUM_LAYERS = 16
NUM_EXPERTS = 64
CHECKPOINT_NAME = "checkpoint_70"

# Target Geometry Construction
ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.50, 2.00]
TARGET_ANGLES_DEG = [0, 15, 30, 45, 60] # plus theta_max_i generated dynamically

# Intervention Hyperparameters
UPDATE_STEPS = 50
LR = 5e-4
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 2
SEEDS = [42, 100, 2024] # For Control E (Variance)

# Calibration Configurations
CALIBRATION_CONFIGS = [
    {"name": "A", "steps": 5, "lr": 1e-4},
    {"name": "B", "steps": 25, "lr": 1e-4},
    {"name": "C", "steps": 50, "lr": 1e-4},
    {"name": "D", "steps": 50, "lr": 5e-4},
]

def dump_config():
    ensure_dirs()
    config_dict = {
        "MODEL_ID": MODEL_ID,
        "CHECKPOINT_NAME": CHECKPOINT_NAME,
        "ALPHAS": ALPHAS,
        "TARGET_ANGLES_DEG": TARGET_ANGLES_DEG,
        "UPDATE_STEPS": UPDATE_STEPS,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "MICRO_BATCH_SIZE": MICRO_BATCH_SIZE,
        "SEEDS": SEEDS,
        "CALIBRATION_CONFIGS": CALIBRATION_CONFIGS
    }
    with open(os.path.join(DIRS["results"], "EXP6D_CONFIG.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

if __name__ == "__main__":
    dump_config()
