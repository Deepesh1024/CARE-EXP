"""
CARE-MoE Experiment 3B — Central Configuration
=================================================
All constants, paths, and parameters for Capability Geometry Validation Phase A.

This experiment uses ONLY ground-truth Oracle KL from Experiment 1.
No XGBoost predictions. No surrogate-derived geometry.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Input data (shared — READ ONLY)
DATA_PATH = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")

# Experiment 3B output
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp3b")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Dataset Filtering
# ──────────────────────────────────────────────
# Use the most reliable calibration size (Exp 1 / 1.5 conclusion)
SEQ_LEN_FILTER = 512

# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────
N_EXPERTS = 64
TOTAL_MOE_LAYERS = 16
LAYERS = ["first", "middle", "last"]

# Dimensions to evaluate for capability geometry
Q_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Cross-Validation Configuration
# ──────────────────────────────────────────────
N_FOLDS = 5                 # Expert-level CV folds
N_REPETITIONS = 10          # Independent shuffle repetitions for robustness

# SMACOF parameters
SMACOF_MAX_ITER = 3000
SMACOF_N_INIT = 4           # Number of random initializations
SMACOF_EPS = 1e-4           # Convergence tolerance
SMACOF_METRIC = True        # Metric MDS (SMACOF)

# Out-of-sample embedding optimization
OOS_N_RESTARTS = 5          # Number of random initializations for held-out embedding
OOS_OPTIM_METHOD = "L-BFGS-B"
OOS_OPTIM_MAXITER = 500

# ──────────────────────────────────────────────
# Null Model Configuration
# ──────────────────────────────────────────────
N_NULL_REALIZATIONS = 30   # Number of null realizations for stable CIs

# Null B: Random Euclidean
NULL_B_DIM = 64             # Ambient dimension for random Euclidean null
NULL_B_N_POINTS = 64        # Number of random points (matching N_EXPERTS)

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# ──────────────────────────────────────────────
# Numerical Stability
# ──────────────────────────────────────────────
EPSILON = 1e-10
