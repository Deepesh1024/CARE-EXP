"""
CARE-MoE Experiment 3A — Central Configuration
=================================================
All constants, paths, and parameters for Capability Graph Discovery.

This experiment uses ONLY frozen outputs from Experiments 1 and 2.
No retraining. No feature engineering. No oracle recomputation.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Input data (shared — READ ONLY)
DATA_PATH = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")

# Frozen Experiment 2 surrogate — READ ONLY, NEVER OVERWRITE
EXP2_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp2")
EXP2_MODELS_DIR = os.path.join(EXP2_RESULTS_DIR, "models")
FROZEN_SURROGATE_PATH = os.path.join(EXP2_MODELS_DIR, "XGBoost_C.pkl")
FROZEN_SCALER_PATH = os.path.join(EXP2_MODELS_DIR, "scaler.pkl")

# Experiment 3A output
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp3a")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
COMMUNITIES_DIR = os.path.join(RESULTS_DIR, "communities")
BASELINES_DIR = os.path.join(RESULTS_DIR, "baselines")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Key output files
REPORT_PATH = os.path.join(RESULTS_DIR, "experiment3a_report.md")
PRE_REGISTRATION_PATH = os.path.join(RESULTS_DIR, "pre_registration.md")

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Dataset Filtering
# ──────────────────────────────────────────────
# Use the most reliable calibration size (Exp 1 conclusion)
SEQ_LEN_FILTER = 512

# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────
N_EXPERTS = 64
TOTAL_MOE_LAYERS = 16
LAYERS = ["first", "middle", "last"]

LAYER_DEPTH_MAP = {
    "first":  0 / (TOTAL_MOE_LAYERS - 1),       # 0.0
    "middle": (TOTAL_MOE_LAYERS // 2) / (TOTAL_MOE_LAYERS - 1),  # ~0.533
    "last":   1.0,
}

# ──────────────────────────────────────────────
# Feature Definitions (must match frozen Exp 2 surrogate)
# ──────────────────────────────────────────────
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

ALL_FEATURES = ORIGINAL_FEATURES + NEW_DESCRIPTORS

# ──────────────────────────────────────────────
# Graph Construction Parameters
# ──────────────────────────────────────────────
K_VALUES = [5, 8, 10]          # Mutual-kNN neighbourhood sizes
K_PRIMARY = 8                  # Primary analysis

# Random baseline
N_RANDOM_GRAPHS = 1000         # Number of random graphs per comparison

# ──────────────────────────────────────────────
# Numerical Stability
# ──────────────────────────────────────────────
EPSILON = 1e-10

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
