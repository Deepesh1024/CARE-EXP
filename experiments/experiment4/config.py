"""
CARE-MoE Experiment 4 — Central Configuration
===============================================
All constants, paths, feature definitions, and hyperparameters.

IMMUTABLE after experiment_config.json is written.

Scientific question:
    Does capability geometry contain predictive information about
    functional merge damage that is not captured by existing local
    pre-merge descriptors?

Three models:
    A: Local pre-merge features only (XGBoost, retrained per fold)
    B: Capability geometry only — prediction = ||z_i - z_j||_2 (NOT learned)
    C: Local features + geometry distance (XGBoost, retrained per fold)

Layer: middle only.
Conclusions must be explicitly labeled "middle-layer-only".
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Input: Oracle distance matrix from Experiment 3B (validated, READ ONLY)
ORACLE_MATRIX_PATH = os.path.join(
    _PROJECT_ROOT, "results", "exp3b", "oracle_distance_matrix_middle.csv"
)

# Input: Raw feature data from Experiment 1 (READ ONLY)
RAW_DATA_PATH = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")

# Output: Experiment 4 results
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp4")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
NOISE_CEILING_DIR = os.path.join(RESULTS_DIR, "noise_ceiling")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Key output files
EXPERIMENT_CONFIG_PATH = os.path.join(RESULTS_DIR, "experiment_config.json")
CV_SPLITS_PATH = os.path.join(RESULTS_DIR, "cv_splits.json")
FEATURE_PROVENANCE_PATH = os.path.join(RESULTS_DIR, "feature_provenance.json")
FOLD_METRICS_CSV = os.path.join(RESULTS_DIR, "fold_metrics.csv")
PARTITION_METRICS_CSV = os.path.join(RESULTS_DIR, "partition_metrics.csv")
FINAL_REPORT_JSON = os.path.join(RESULTS_DIR, "final_report.json")
FINAL_REPORT_MD = os.path.join(RESULTS_DIR, "final_report.md")

# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────
N_EXPERTS = 64
LAYER = "middle"
N_PAIRS = 2016  # C(64, 2)

# Seq_Len for raw feature extraction.
# MUST match the calibration configuration of Exp 3B Oracle matrix (Seq_Len=512).
# Avoids calibration-distribution mismatch.
SEQ_LEN = 512

# ──────────────────────────────────────────────
# Capability Geometry
# ──────────────────────────────────────────────
# q=4 was selected in Exp 3B as best-performing among q=2,4,6,8.
# It was PERFORMANCE-SELECTED, not theoretically motivated.
# q is FIXED here — must NOT be re-tuned based on Experiment 4 results.
Q = 4

# MDS parameters (matching Exp 3B exactly)
SMACOF_MAX_ITER = 3000
SMACOF_N_INIT = 5           # MDS restarts — fixed before execution
SMACOF_EPS = 1e-4
SMACOF_METRIC = True        # Metric MDS (SMACOF)

# Out-of-sample embedding — fixed before execution
OOS_N_RESTARTS = 5          # Test-point embedding restarts
OOS_OPTIM_METHOD = "L-BFGS-B"
OOS_OPTIM_MAXITER = 500

# ──────────────────────────────────────────────
# Expert-Disjoint Cross Validation
# ──────────────────────────────────────────────
N_PARTITIONS = 5
N_FOLDS = 3

# Five fixed deterministic partition seeds — NEVER change after run begins
PARTITION_SEEDS = [1001, 2002, 3003, 4004, 5005]

# Pilot: run only first two partitions (integrity test only)
PILOT_PARTITIONS = 2

# ──────────────────────────────────────────────
# Local Features
# ──────────────────────────────────────────────
# 7 original pre-merge features directly stored in Exp 1 output.json.
# These are PAIR-LOCAL: computed solely from expert pair (i, j) statistics.
ORIGINAL_FEATURES = [
    "Weight_Distance",       # pair-local
    "Weight_Cosine",         # pair-local
    "Activation_Similarity", # pair-local
    "Output_Similarity",     # pair-local
    "Routing_Similarity",    # pair-local
    "Usage_Frequency",       # pair-local (|A∪B|/N)
    "Jaccard_Overlap",       # pair-local
]

# 4 new CARE capability descriptors from Exp 2.
# Some use per-expert MARGINAL statistics computed across all pairs containing
# that expert — these are flagged as using global statistics.
# See FEATURE_PROVENANCE table for full audit.
NEW_DESCRIPTORS = [
    "Usage_Asymmetry",       # FLAGGED: uses per-expert global marginal usage
    "Routing_JSD_Proxy",     # pair-local (derived from Routing_Similarity, Jaccard_Overlap)
    "Routing_NPMI_Proxy",    # FLAGGED: uses per-expert global marginal usage + global mean
    "Specialization_Diff",   # FLAGGED: uses per-expert global marginal usage
]

# All 11 local features for Model A (and as base for Model C)
LOCAL_FEATURES = ORIGINAL_FEATURES + NEW_DESCRIPTORS

# Geometry feature name (for Model B/C)
# Model B prediction = ||z_i - z_j||_2 — NOT a learned predictor
GEOMETRY_FEATURE = "Geometry_Distance"

# Columns that must NEVER be used as features (leakage prevention)
FORBIDDEN_FEATURES = [
    "Oracle_KL",
    "Oracle_KL_SplitA",
    "Oracle_KL_SplitB",
    "Runtime_Sec",
    "Max_VRAM_MB",
    "Random_Baseline",
    "CrossEntropy_Delta",      # post-merge
    "Hidden_L2_Drift",         # post-merge
    "Router_Entropy_Orig",     # may contain post-merge info
    "Router_Entropy_Merged",   # post-merge
    "Top1_Routing_Agreement",  # post-merge
    "TopK_Routing_Agreement",  # post-merge
]

# Numerical stability
EPSILON = 1e-10

# ──────────────────────────────────────────────
# Model Hyperparameters (identical to Exp 2)
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# XGBoost — identical to Exp 2 config.py
# If XGBoost is unavailable, experiment STOPS and reports incompatibility.
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
# Statistical Thresholds (PRE-REGISTERED)
# ──────────────────────────────────────────────
# Minimum practically meaningful Spearman rho improvement.
# FIXED before execution — DO NOT MODIFY after observing results.
DELTA_RHO_MIN = 0.05

# Bootstrap CI over 5 partition-level effects
BOOTSTRAP_N_SAMPLES = 10_000
BOOTSTRAP_CI_LEVEL = 0.95

# Statistical power note (pre-registered, not post-hoc):
# With only 5 independent partition-level observations, bootstrap CIs
# reflect sampling variability across 5 units and must NOT be interpreted
# as high-power inference. CIs are reported for completeness.

# Precision@K evaluation points (absolute K, not percentage)
PRECISION_K_VALUES = [10, 25, 50]

# ──────────────────────────────────────────────
# Noise Ceiling
# ──────────────────────────────────────────────
# Status: SKIPPED unless genuine repeated Oracle measurements are provided.
# Do NOT use multi-Seq_Len values as independent replicates —
# they are NOT independent repeated measurements of the same quantity.
NOISE_CEILING_STATUS = "SKIPPED"
NOISE_CEILING_REASON = (
    "No genuine repeated Oracle measurements available. "
    "Multi-Seq_Len values are not independent replicates. "
    "Noise ceiling will be reported as SKIPPED."
)

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# ──────────────────────────────────────────────
# Code Version Identifier
# ──────────────────────────────────────────────
EXPERIMENT_VERSION = "experiment4-v1.1"
