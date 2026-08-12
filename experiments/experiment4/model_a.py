"""
CARE-MoE Experiment 4 — Model A: Local Feature Baseline
=========================================================
XGBoost trained from scratch on local pre-merge features per fold.

Spec §7:
  - Retrained independently inside every fold.
  - Never uses held-out experts during training.
  - Uses identical XGBoost hyperparameters as Experiment 2.
  - RobustScaler fitted on training pairs only.

If XGBoost is unavailable, STOP — do NOT substitute silently.
"""

import os
import sys

import numpy as np
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import XGBOOST_PARAMS, LOCAL_FEATURES, RANDOM_SEED

# Import XGBoost — hard stop if unavailable
try:
    from xgboost import XGBRegressor
    _ENGINE = "XGBoost"
except ImportError:
    raise ImportError(
        "STOP: XGBoost is not installed. "
        "Experiment 2 used XGBoost; Experiment 4 must reproduce that configuration. "
        "Install xgboost and retry. Do NOT substitute another model."
    )


def train_model_a(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """Train Model A (local features only) from scratch.

    Parameters
    ----------
    X_train : (n_train_pairs, 11) unscaled local features.
    y_train : (n_train_pairs,) Oracle KL targets.

    Returns
    -------
    model    : fitted XGBRegressor
    scaler   : fitted RobustScaler (fit on X_train only)
    """
    assert X_train.shape[1] == len(LOCAL_FEATURES), (
        f"Expected {len(LOCAL_FEATURES)} features, got {X_train.shape[1]}"
    )
    assert len(X_train) == len(y_train)
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train))

    # Scale: fit on training pairs only
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    # Train from scratch
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_model_a(
    model,
    scaler,
    X_test: np.ndarray,
) -> np.ndarray:
    """Predict with Model A on test pairs.

    Parameters
    ----------
    X_test : (n_test_pairs, 11) unscaled local features.

    Returns
    -------
    y_pred : (n_test_pairs,) predicted Oracle KL.
    """
    assert X_test.shape[1] == len(LOCAL_FEATURES)
    X_test_scaled = scaler.transform(X_test.astype(np.float64))
    y_pred = model.predict(X_test_scaled)
    return y_pred.astype(np.float32)
