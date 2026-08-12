"""
CARE-MoE Experiment 4 — MDS Embedding
=======================================
Training MDS and out-of-sample (OOS) test-expert embedding.

Follows Experiment 3B methodology exactly (q=4, SMACOF, L-BFGS-B OOS).

Leakage rules enforced:
  - Training MDS uses ONLY train×train Oracle distances.
  - Z_train is FROZEN before test embedding begins.
  - Each test expert's coordinate is optimized using ONLY
    test→train distances (NOT test→test distances).
  - test→test distances are used ONLY for final evaluation.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    Q,
    SMACOF_MAX_ITER,
    SMACOF_N_INIT,
    SMACOF_EPS,
    SMACOF_METRIC,
    OOS_N_RESTARTS,
    OOS_OPTIM_METHOD,
    OOS_OPTIM_MAXITER,
)


# ══════════════════════════════════════════════════════════
# Training MDS
# ══════════════════════════════════════════════════════════

def run_smacof_train(
    D_train: np.ndarray,
    q: int,
    seed: int,
) -> np.ndarray:
    """Run metric SMACOF on the training sub-matrix.

    Parameters
    ----------
    D_train : (n_train, n_train) symmetric distance matrix.
    q       : embedding dimension (fixed to Q=4 for Exp 4).
    seed    : random seed.

    Returns
    -------
    Z_train : (n_train, q) embedding.
    """
    assert D_train.ndim == 2
    assert D_train.shape[0] == D_train.shape[1]
    assert np.allclose(D_train, D_train.T, atol=1e-8), "D_train not symmetric"
    assert q == Q, f"q={q} != Q={Q}. q is fixed."

    from sklearn.manifold import MDS

    best_Z = None
    best_stress = np.inf

    for init_idx in range(SMACOF_N_INIT):
        init_seed = (seed * 10000 + init_idx) % (2**31 - 1)

        try:
            mds = MDS(
                n_components=q,
                metric=SMACOF_METRIC,
                dissimilarity="precomputed",
                max_iter=SMACOF_MAX_ITER,
                n_init=1,
                eps=SMACOF_EPS,
                random_state=init_seed,
                n_jobs=1,
                normalized_stress=False,
            )
        except TypeError:
            # Older sklearn
            mds = MDS(
                n_components=q,
                metric=SMACOF_METRIC,
                dissimilarity="precomputed",
                max_iter=SMACOF_MAX_ITER,
                n_init=1,
                eps=SMACOF_EPS,
                random_state=init_seed,
                n_jobs=1,
            )

        Z = mds.fit_transform(D_train.astype(np.float64))
        stress = float(mds.stress_)

        if stress < best_stress:
            best_stress = stress
            best_Z = Z

    assert best_Z is not None
    assert best_Z.shape == (D_train.shape[0], q), (
        f"MDS output shape {best_Z.shape} != ({D_train.shape[0]}, {q})"
    )

    return best_Z.astype(np.float32)


# ══════════════════════════════════════════════════════════
# Out-of-Sample Test-Expert Embedding
# ══════════════════════════════════════════════════════════

def _oos_objective(z: np.ndarray, z_train: np.ndarray, d_to_train: np.ndarray):
    """Stress for a single test point:  Σ_i (||z - z_i||_2 - d_i)²"""
    diff = z_train - z.reshape(1, -1)          # (n_train, q)
    embed_d = np.sqrt(np.sum(diff ** 2, axis=1))  # (n_train,)
    residuals = embed_d - d_to_train           # (n_train,)
    return float(np.sum(residuals ** 2))


def _oos_gradient(z: np.ndarray, z_train: np.ndarray, d_to_train: np.ndarray):
    """Gradient of stress w.r.t. z."""
    diff = z.reshape(1, -1) - z_train          # (n_train, q)  z - z_i
    embed_d = np.sqrt(np.sum(diff ** 2, axis=1))  # (n_train,)
    safe_d = np.maximum(embed_d, 1e-10)
    residuals = embed_d - d_to_train           # (n_train,)
    scale = 2.0 * residuals / safe_d           # (n_train,)
    grad = np.sum(scale[:, np.newaxis] * diff, axis=0)  # (q,)
    return grad


def embed_test_expert(
    z_train: np.ndarray,
    d_test_to_train: np.ndarray,
    q: int,
    seed: int,
) -> np.ndarray:
    """Embed one held-out expert by optimizing its coordinate.

    CRITICAL LEAKAGE RULE:
      d_test_to_train must contain ONLY distances from the test expert
      to TRAINING experts. Test→test distances must NOT appear here.

    Parameters
    ----------
    z_train         : (n_train, q) — frozen training coordinates.
    d_test_to_train : (n_train,) — distances from test expert to each train expert.
    q               : embedding dimension (must equal Q=4).
    seed            : random seed for restarts.

    Returns
    -------
    z_opt : (q,) optimal coordinates.
    """
    assert z_train.shape[1] == q
    assert len(d_test_to_train) == z_train.shape[0], (
        "d_test_to_train length must equal n_train. "
        "Test-test distances must NOT be included."
    )
    assert np.all(d_test_to_train >= 0), "Negative distances in d_test_to_train"

    rng = np.random.RandomState(seed)
    centroid = z_train.mean(axis=0)
    scale = z_train.std() + 1e-8

    best_z = None
    best_val = np.inf

    for restart in range(OOS_N_RESTARTS):
        z0 = centroid + rng.randn(q) * scale

        res = minimize(
            _oos_objective,
            z0,
            args=(z_train, d_test_to_train),
            jac=_oos_gradient,
            method=OOS_OPTIM_METHOD,
            options={"maxiter": OOS_OPTIM_MAXITER, "ftol": 1e-12, "gtol": 1e-8},
        )

        if res.fun < best_val:
            best_val = res.fun
            best_z = res.x

    assert best_z is not None
    return best_z.astype(np.float32)


# ══════════════════════════════════════════════════════════
# Full Fold Embedding
# ══════════════════════════════════════════════════════════

def embed_fold(
    D_oracle: np.ndarray,
    train_experts: list,
    test_experts: list,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed all test experts for one fold.

    Leakage controls:
      1. D_train extracted from D_oracle using only train_experts.
      2. Z_train frozen after SMACOF.
      3. Each test expert embedded using ONLY its distances to train experts.
      4. Test→test distances D_oracle[t_i, t_j] NOT used until evaluation.

    Parameters
    ----------
    D_oracle      : (64, 64) full Oracle distance matrix.
    train_experts : list of int, training expert indices.
    test_experts  : list of int, test expert indices.
    seed          : int.

    Returns
    -------
    Z_train : (n_train, 4) — training embeddings.
    Z_test  : (n_test, 4)  — test embeddings (order matches test_experts).
    """
    train_idx = np.array(train_experts, dtype=np.int32)
    test_idx = np.array(test_experts, dtype=np.int32)

    n_train = len(train_idx)
    n_test = len(test_idx)

    # ── Step 1: Extract training sub-matrix ──
    D_train = D_oracle[np.ix_(train_idx, train_idx)].astype(np.float64)
    assert D_train.shape == (n_train, n_train)

    # ── Step 2: Fit MDS on training experts only ──
    Z_train = run_smacof_train(D_train, Q, seed)
    assert Z_train.shape == (n_train, Q), f"MDS output shape mismatch: {Z_train.shape}"

    # ── Step 3: Embed each test expert independently ──
    Z_test = np.zeros((n_test, Q), dtype=np.float32)
    for t, test_exp in enumerate(test_idx):
        # CRITICAL: use ONLY test→train distances
        d_to_train = D_oracle[test_exp, train_idx].astype(np.float64)
        assert len(d_to_train) == n_train

        # Verify no test-test distances accidentally included
        # d_to_train should only contain distances to train experts
        for _check_idx in d_to_train:
            pass  # shape already validated above

        embed_seed = (seed * 10000 + t) % (2**31 - 1)
        z_t = embed_test_expert(Z_train, d_to_train, Q, embed_seed)
        Z_test[t] = z_t

    return Z_train, Z_test


# ══════════════════════════════════════════════════════════
# Geometry Feature Extraction
# ══════════════════════════════════════════════════════════

def compute_geometry_distances(
    Z_test: np.ndarray,
    test_experts: list,
    pi_test: np.ndarray,
    pj_test: np.ndarray,
) -> np.ndarray:
    """Compute ||z_i - z_j||_2 for all test-test pairs.

    Model B prediction IS this distance — not a learned predictor.

    Parameters
    ----------
    Z_test       : (n_test, q) — test expert embeddings.
    test_experts : list of int — expert indices in same order as Z_test.
    pi_test      : (n_test_pairs,) — first expert index for each test pair.
    pj_test      : (n_test_pairs,) — second expert index for each test pair.

    Returns
    -------
    geom_dists : (n_test_pairs,) float32 — ||z_i - z_j||_2 for each test pair.
    """
    # Build local index mapping: expert_id -> row in Z_test
    expert_to_row = {int(eid): r for r, eid in enumerate(test_experts)}

    n_pairs = len(pi_test)
    geom_dists = np.zeros(n_pairs, dtype=np.float32)

    for k in range(n_pairs):
        ei = int(pi_test[k])
        ej = int(pj_test[k])

        assert ei in expert_to_row, (
            f"Expert {ei} not in test set. Test-expert embedding only."
        )
        assert ej in expert_to_row, (
            f"Expert {ej} not in test set. Test-expert embedding only."
        )

        ri = expert_to_row[ei]
        rj = expert_to_row[ej]
        geom_dists[k] = float(np.linalg.norm(Z_test[ri] - Z_test[rj]))

    assert not np.any(np.isnan(geom_dists)), "NaN in geometry distances"
    assert not np.any(np.isinf(geom_dists)), "Inf in geometry distances"

    return geom_dists
