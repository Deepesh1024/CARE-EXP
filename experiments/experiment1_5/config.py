"""
CARE-MoE Experiment 1.5 — Central Configuration
=================================================
All constants, paths, feature definitions, and hyperparameters used across
Phase 1 (dataset), Phase 2 (regression), and Phase 3 (analysis).

Every script in this experiment imports from here — nothing is hardcoded
in individual phase files.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

DATA_PATH = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp1_5")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

TRAIN_PARQUET = os.path.join(RESULTS_DIR, "train_df.parquet")
TEST_PARQUET = os.path.join(RESULTS_DIR, "test_df.parquet")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "experiment1_5_report.md")

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Dataset Filtering
# ──────────────────────────────────────────────
# Use the largest calibration size — most reliable oracle signal and
# the only slice with split-sample stability data.
SEQ_LEN_FILTER = 256

# ──────────────────────────────────────────────
# Expert Split
# ──────────────────────────────────────────────
TRAIN_EXPERTS = set(range(0, 32))
TEST_EXPERTS = set(range(32, 64))

# ──────────────────────────────────────────────
# Target
# ──────────────────────────────────────────────
TARGET = "Oracle_KL"

# ──────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────
# Only genuinely pre-merge proxy features.
# Oracle-only features (CrossEntropy_Delta, Hidden_L2_Drift,
# Router_Entropy_Orig/Merged, Top1/TopK_Routing_Agreement)
# are excluded — they require the merge + second forward pass
# and therefore cannot serve as cheap predictors.
FEATURES = [
    "Weight_Distance",
    "Weight_Cosine",
    "Activation_Similarity",
    "Output_Similarity",
    "Routing_Similarity",
    "Usage_Frequency",
    "Jaccard_Overlap",
]

# Columns that must never be used as features
EXCLUDE_COLS = [
    "Oracle_KL",
    "Oracle_KL_SplitA",
    "Oracle_KL_SplitB",
    "Runtime_Sec",
    "Max_VRAM_MB",
    "Random_Baseline",
    # Oracle-only features (require merge + second forward pass)
    "CrossEntropy_Delta",
    "Hidden_L2_Drift",
    "Router_Entropy_Orig",
    "Router_Entropy_Merged",
    "Top1_Routing_Agreement",
    "TopK_Routing_Agreement",
]

# ──────────────────────────────────────────────
# Layer Depth Mapping
# ──────────────────────────────────────────────
# OLMoE-1B-7B has 16 MoE layers (indices 0–15).
# Experiment 1 used: first=0, middle=len//2, last=len-1.
TOTAL_MOE_LAYERS = 16

LAYER_DEPTH_MAP = {
    "first":  0  / (TOTAL_MOE_LAYERS - 1),   # 0.0
    "middle": (TOTAL_MOE_LAYERS // 2) / (TOTAL_MOE_LAYERS - 1),  # ~0.533
    "last":   1.0,
}

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 1e-4       # small α so LASSO doesn't kill everything on a small dataset
LASSO_MAX_ITER = 50_000

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
