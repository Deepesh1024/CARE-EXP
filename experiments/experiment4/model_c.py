"""
CARE-MoE Experiment 4 — Model C: CARE (Local + Geometry)
==========================================================
Model C is the pre-registered CARE candidate.

Features: 11 local features + 1 geometry distance = 12 features total.

Spec §9:
  - Trained from scratch within every fold.
  - Uses identical XGBoost hyperparameters as Model A / Experiment 2.
  - RobustScaler fitted on training pairs only (12 features).
  - Geometry distance for TEST pairs from OOS embedding.
  - Geometry distance for TRAIN pairs from training MDS embedding.

Excluded features (as per spec §9):
  - No degree, centrality, PageRank, graph density, Louvain, etc.
  - No Jacobian, Hessian, metric tensor.
  - No hand-selected post-hoc features.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import XGBOOST_PARAMS, LOCAL_FEATURES, GEOMETRY_FEATURE, RANDOM_SEED

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError(
        "STOP: XGBoost not installed. "
        "Do NOT substitute another model for Model C."
    )

MODEL_C_FEATURES = list(LOCAL_FEATURES) + [GEOMETRY_FEATURE]


def compute_train_geometry(
    Z_train: np.ndarray,
    pi_train: np.ndarray,
    pj_train: np.ndarray,
    train_experts: list,
) -> np.ndarray:
    """Compute ||z_i - z_j||_2 for train-train pairs.

    Uses Z_train (training MDS coordinates only).

    Parameters
    ----------
    Z_train      : (n_train_experts, q) training embeddings.
    pi_train     : (n_train_pairs,) first expert index per pair.
    pj_train     : (n_train_pairs,) second expert index per pair.
    train_experts: list of expert IDs (same order as Z_train rows).

    Returns
    -------
    geom_train : (n_train_pairs,) float32
    """
    expert_to_row = {int(eid): r for r, eid in enumerate(train_experts)}
    n = len(pi_train)
    geom = np.zeros(n, dtype=np.float32)
    for k in range(n):
        ri = expert_to_row[int(pi_train[k])]
        rj = expert_to_row[int(pj_train[k])]
        geom[k] = float(np.linalg.norm(Z_train[ri] - Z_train[rj]))
    return geom


def train_model_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    geom_train: np.ndarray,
) -> tuple:
    """Train Model C (local + geometry) from scratch.

    Parameters
    ----------
    X_train    : (n_train_pairs, 11) unscaled local features.
    y_train    : (n_train_pairs,) Oracle KL targets.
    geom_train : (n_train_pairs,) geometry distances for training pairs
                 computed from Z_train (training MDS coordinates).

    Returns
    -------
    model  : fitted XGBRegressor
    scaler : fitted RobustScaler (12 features, fit on training pairs only)
    """
    assert X_train.shape[1] == len(LOCAL_FEATURES)
    assert len(geom_train) == len(X_train)
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(geom_train))
    assert not np.any(np.isnan(y_train))

    # Concatenate: 11 local + 1 geometry = 12 features
    X_c = np.column_stack([X_train.astype(np.float64),
                            geom_train.reshape(-1, 1).astype(np.float64)])
    assert X_c.shape[1] == len(MODEL_C_FEATURES), (
        f"Expected {len(MODEL_C_FEATURES)} features, got {X_c.shape[1]}"
    )

    # Scale: fit on training pairs only
    scaler = RobustScaler()
    X_c_scaled = scaler.fit_transform(X_c)

    # Train from scratch
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_c_scaled, y_train)

    return model, scaler


def predict_model_c(
    model,
    scaler,
    X_test: np.ndarray,
    geom_test: np.ndarray,
) -> np.ndarray:
    """Predict with Model C on test pairs.

    Parameters
    ----------
    X_test    : (n_test_pairs, 11) unscaled local features.
    geom_test : (n_test_pairs,) geometry distances for test pairs
                (||z_i - z_j||_2 from OOS embeddings).

    Returns
    -------
    y_pred : (n_test_pairs,) float32
    """
    assert X_test.shape[1] == len(LOCAL_FEATURES)
    assert len(geom_test) == len(X_test)

    X_c = np.column_stack([X_test.astype(np.float64),
                            geom_test.reshape(-1, 1).astype(np.float64)])
    X_c_scaled = scaler.transform(X_c)
    y_pred = model.predict(X_c_scaled)
    return y_pred.astype(np.float32)
