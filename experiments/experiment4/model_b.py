"""
CARE-MoE Experiment 4 — Model B: Geometry Only
================================================
Model B is NOT a learned predictor.

Prediction is simply:

    prediction_B(i,j) = ||z_i - z_j||_2

where z_i, z_j are the MDS embeddings of test experts i and j
in the q=4 capability space.

Spec §8, User correction §5:
  - No model is trained for Model B.
  - The geometry distance IS the prediction.
  - Evaluate Spearman, Pearson, RMSE, MAE against Oracle KL targets.
"""

import numpy as np


def predict_model_b(geom_dists: np.ndarray) -> np.ndarray:
    """Return geometry distances as Model B predictions.

    Parameters
    ----------
    geom_dists : (n_test_pairs,) — precomputed ||z_i - z_j||_2.

    Returns
    -------
    y_pred : (n_test_pairs,) float32 — same as geom_dists (no transformation).
    """
    assert not np.any(np.isnan(geom_dists)), "NaN in geometry distances"
    assert not np.any(np.isinf(geom_dists)), "Inf in geometry distances"
    assert np.all(geom_dists >= 0), "Negative geometry distances"
    return geom_dists.astype(np.float32)
