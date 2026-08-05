"""
CARE-MoE Experiment 2 — Shared Utilities
============================================
Data loading, directory scaffolding, seed management, persistence helpers,
and publication-quality plotting setup. Extends Exp 1.5 utilities with
CSV export and deeper directory management.
"""

import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_PATH,
    RESULTS_DIR,
    PLOTS_DIR,
    TABLES_DIR,
    MODELS_DIR,
    PLOT_RESIDUALS_DIR,
    PLOT_CORRELATIONS_DIR,
    PLOT_DESCRIPTOR_DIR,
    PLOT_REGRESSION_DIR,
    PLOT_SHAP_DIR,
    PLOT_ABLATION_DIR,
    RANDOM_SEED,
    FIGURE_DPI,
    FIGURE_FORMAT,
)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────
# Directory Management
# ──────────────────────────────────────────────
_ALL_DIRS = [
    RESULTS_DIR, PLOTS_DIR, TABLES_DIR, MODELS_DIR,
    PLOT_RESIDUALS_DIR, PLOT_CORRELATIONS_DIR, PLOT_DESCRIPTOR_DIR,
    PLOT_REGRESSION_DIR, PLOT_SHAP_DIR, PLOT_ABLATION_DIR,
]


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    """Load the Experiment 1 output JSON and return the results DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per (Seq_Len, Layer, Expert_A, Expert_B) evaluation.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Ensure Experiment 1 output.json exists."
        )
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        raise ValueError("The 'results' key in output.json is empty.")

    df = pd.DataFrame(results)
    print(f"[utils] Loaded {len(df):,} rows from {DATA_PATH}")
    return df


# ──────────────────────────────────────────────
# Persistence Helpers
# ──────────────────────────────────────────────
def save_pickle(obj, path: str) -> None:
    """Serialize an arbitrary Python object to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[utils] Saved pickle → {path}")


def load_pickle(path: str):
    """Deserialize a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str) -> None:
    """Write a JSON-serializable object to disk with pretty formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"[utils] Saved JSON → {path}")


def load_json(path: str):
    """Read a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[utils] Saved CSV → {path}")


# ──────────────────────────────────────────────
# Publication-Quality Plotting
# ──────────────────────────────────────────────
def set_pub_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, name: str, subdir: str = "") -> str:
    """Save a figure to the appropriate plots subdirectory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename without extension.
    subdir : str
        Subdirectory under PLOTS_DIR (e.g., 'residuals', 'shap').

    Returns
    -------
    str : path to saved figure.
    """
    target_dir = os.path.join(PLOTS_DIR, subdir) if subdir else PLOTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{name}.{FIGURE_FORMAT}")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Figure → {path}")
    return path
