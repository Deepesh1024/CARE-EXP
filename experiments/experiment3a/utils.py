"""
CARE-MoE Experiment 3A — Shared Utilities
============================================
Data loading, directory scaffolding, seed management, persistence helpers,
and publication-quality plotting setup.
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
    GRAPHS_DIR,
    COMMUNITIES_DIR,
    BASELINES_DIR,
    VALIDATION_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    RANDOM_SEED,
    FIGURE_DPI,
    FIGURE_FORMAT,
)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────
# Directory Management
# ──────────────────────────────────────────────
_ALL_DIRS = [
    RESULTS_DIR, GRAPHS_DIR, COMMUNITIES_DIR,
    BASELINES_DIR, VALIDATION_DIR, FIGURES_DIR, TABLES_DIR,
]


def ensure_dirs() -> None:
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}.")
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[utils] Saved pickle → {path}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return sorted(list(o))
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"[utils] Saved JSON → {path}")


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[utils] Saved CSV → {path}")


# ──────────────────────────────────────────────
# Publication-Quality Plotting
# ──────────────────────────────────────────────
def set_pub_style() -> None:
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


def save_fig(fig, name: str) -> str:
    path = os.path.join(FIGURES_DIR, f"{name}.{FIGURE_FORMAT}")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Figure → {path}")
    return path
