"""
CARE-MoE Experiment 2 — Central Configuration
=================================================
All constants, paths, feature definitions, and hyperparameters.

Inherits Experiment 1.5 conventions:
  - Same RANDOM_SEED (42)
  - Same SEQ_LEN_FILTER (256)
  - Same expert split boundaries (0–31 train, 32–63 test)
  - Same model hyperparameters (Ridge α, LASSO α, XGBoost params)
  - Same RobustScaler normalization

Extends with:
  - 4 new CARE capability descriptors
  - Results directory tree for Experiment 2
  - Reference to frozen Experiment 1.5 baseline
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Input data (shared with Exp 1 / 1.5)
DATA_PATH = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")

# Frozen Experiment 1.5 baseline — READ ONLY, NEVER OVERWRITE
EXP15_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp1_5")
EXP15_METRICS_PATH = os.path.join(EXP15_RESULTS_DIR, "metrics.json")
EXP15_TRAIN_PARQUET = os.path.join(EXP15_RESULTS_DIR, "train_df.parquet")
EXP15_TEST_PARQUET = os.path.join(EXP15_RESULTS_DIR, "test_df.parquet")

# Experiment 2 output
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "exp2")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Plot subdirectories
PLOT_RESIDUALS_DIR = os.path.join(PLOTS_DIR, "residuals")
PLOT_CORRELATIONS_DIR = os.path.join(PLOTS_DIR, "correlations")
PLOT_DESCRIPTOR_DIR = os.path.join(PLOTS_DIR, "descriptor_scatter")
PLOT_REGRESSION_DIR = os.path.join(PLOTS_DIR, "regression")
PLOT_SHAP_DIR = os.path.join(PLOTS_DIR, "shap")
PLOT_ABLATION_DIR = os.path.join(PLOTS_DIR, "ablation")

# Key output files
TRAIN_PARQUET = os.path.join(RESULTS_DIR, "train_df.parquet")
TEST_PARQUET = os.path.join(RESULTS_DIR, "test_df.parquet")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# CSV deliverables
FEATURE_STATS_CSV = os.path.join(RESULTS_DIR, "feature_statistics.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(RESULTS_DIR, "feature_importance.csv")
CORRELATION_MATRIX_CSV = os.path.join(RESULTS_DIR, "correlation_matrix.csv")
RESIDUAL_ANALYSIS_CSV = os.path.join(RESULTS_DIR, "residual_analysis.csv")

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Dataset Filtering (identical to Exp 1.5)
# ──────────────────────────────────────────────
SEQ_LEN_FILTER = 256

# ──────────────────────────────────────────────
# Expert Split (identical to Exp 1.5)
# ──────────────────────────────────────────────
TRAIN_EXPERTS = set(range(0, 32))
TEST_EXPERTS = set(range(32, 64))

# ──────────────────────────────────────────────
# Target
# ──────────────────────────────────────────────
TARGET = "Oracle_KL"

# ──────────────────────────────────────────────
# Original Pre-Merge Features (from Exp 1.5)
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

# ──────────────────────────────────────────────
# New CARE Capability Descriptors (Experiment 2)
# ──────────────────────────────────────────────
NEW_DESCRIPTORS = [
    "Usage_Asymmetry",           # |usage_i - usage_j|
    "Routing_JSD_Proxy",         # JSD proxy from routing stats
    "Routing_NPMI_Proxy",        # NPMI proxy from co-activation
    "Specialization_Diff",       # |specialization_i - specialization_j|
]

# Combined feature set
ALL_FEATURES = ORIGINAL_FEATURES + NEW_DESCRIPTORS

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
# Layer Depth Mapping (identical to Exp 1.5)
# ──────────────────────────────────────────────
TOTAL_MOE_LAYERS = 16

LAYER_DEPTH_MAP = {
    "first":  0 / (TOTAL_MOE_LAYERS - 1),       # 0.0
    "middle": (TOTAL_MOE_LAYERS // 2) / (TOTAL_MOE_LAYERS - 1),  # ~0.533
    "last":   1.0,
}

# ──────────────────────────────────────────────
# Model Hyperparameters (identical to Exp 1.5)
# ──────────────────────────────────────────────
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 1e-4
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

# ──────────────────────────────────────────────
# Numerical Stability
# ──────────────────────────────────────────────
EPSILON = 1e-10  # For log/division in JSD/NPMI computations

# ──────────────────────────────────────────────
# Ablation
# ──────────────────────────────────────────────
ABLATION_FEATURES = ALL_FEATURES  # Ablate all 11 features
