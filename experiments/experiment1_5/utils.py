"""
CARE-MoE Experiment 1.5 — Shared Utilities
============================================
Data loading, directory scaffolding, seed management, and persistence
helpers used by every phase script.
"""

import json
import os
import pickle
import random

import numpy as np
import pandas as pd

from config import (
    DATA_PATH,
    RESULTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # sklearn and xgboost read random_state per-estimator;
    # this covers numpy-level randomness used elsewhere.


# ──────────────────────────────────────────────
# Directory Management
# ──────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
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
