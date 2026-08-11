"""
CARE-MoE Experiment 3B — Phase 3: Expert-Level Cross-Validation
=================================================================
1. 5-fold expert holdout × 10 repetitions = 50 total folds.
2. For each fold: SMACOF on training sub-matrix → freeze Z_train.
3. Out-of-sample embedding of held-out experts via coordinate optimization.
4. Evaluate test→train and test→test generalization metrics.
5. Run identically for Oracle, Null A, and Null B.

DATA LEAKAGE PREVENTION:
  - Training embedding uses ONLY train×train distances.
  - Z_train is FROZEN during test embedding.
  - Test embedding uses ONLY test→train distances (not test→test).
  - Test→test evaluation uses held-out Oracle distances not seen during fitting.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import MDS
from sklearn.model_selection import KFold

from config import (
    RESULTS_DIR,
    N_EXPERTS,
    LAYERS,
    Q_VALUES,
    N_FOLDS,
    N_REPETITIONS,
    N_NULL_REALIZATIONS,
    SMACOF_MAX_ITER,
    SMACOF_N_INIT,
    SMACOF_EPS,
    SMACOF_METRIC,
    OOS_N_RESTARTS,
    OOS_OPTIM_METHOD,
    OOS_OPTIM_MAXITER,
    NULL_B_DIM,
    NULL_B_N_POINTS,
    RANDOM_SEED,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_json,
    save_csv,
)

# Import null generators from Phase 2
from phase2_mds_nulls import generate_null_a, generate_null_b


# ──────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────

def run_smacof_train(D_train: np.ndarray, q: int, seed: int) -> np.ndarray:
    """Run SMACOF on training sub-matrix and return embedding Z_train.

    Parameters
    ----------
    D_train : np.ndarray
        Symmetric distance matrix for training experts only (n_train × n_train).
    q : int
        Target dimensionality.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Z_train embedding (n_train × q).
    """
    try:
        mds = MDS(
            n_components=q,
            metric_mds=SMACOF_METRIC,
            metric="precomputed",
            init="random",
            max_iter=SMACOF_MAX_ITER,
            n_init=SMACOF_N_INIT,
            eps=SMACOF_EPS,
            random_state=seed,
            normalized_stress="auto",
            n_jobs=-1,
        )
    except TypeError:
        # Fallback for older scikit-learn versions (e.g. on VM)
        mds = MDS(
            n_components=q,
            metric=SMACOF_METRIC,
            dissimilarity="precomputed",
            max_iter=SMACOF_MAX_ITER,
            n_init=SMACOF_N_INIT,
            eps=SMACOF_EPS,
            random_state=seed,
            n_jobs=-1,
        )
    Z = mds.fit_transform(D_train)
    return Z


def embed_single_test_expert(
    z_train: np.ndarray,
    d_test_to_train: np.ndarray,
    q: int,
    seed: int,
) -> np.ndarray:
    """Embed a single held-out expert by optimizing its coordinates.

    Minimizes: Σ_i (||z - z_i||₂ - d_ji)²
    where z_i are frozen training coordinates and d_ji are known distances.

    Parameters
    ----------
    z_train : np.ndarray
        Frozen training coordinates (n_train × q).
    d_test_to_train : np.ndarray
        Known distances from the test expert to each training expert (n_train,).
    q : int
        Embedding dimensionality.
    seed : int
        Random seed for initializations.

    Returns
    -------
    np.ndarray
        Optimized coordinate z_j (q,).
    """
    rng = np.random.RandomState(seed)

    def objective(z):
        """Stress objective for a single point."""
        z = z.reshape(1, -1)
        embed_dists = np.sqrt(np.sum((z_train - z) ** 2, axis=1))
        residuals = embed_dists - d_test_to_train
        return float(np.sum(residuals ** 2))

    def gradient(z):
        """Gradient of the stress objective."""
        z = z.reshape(1, -1)
        diff = z - z_train  # (n_train, q)
        embed_dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (n_train,)

        # Avoid division by zero
        safe_dists = np.maximum(embed_dists, 1e-10)
        residuals = embed_dists - d_test_to_train  # (n_train,)

        # Gradient: 2 * Σ_i residual_i * (z - z_i) / ||z - z_i||
        scale = 2.0 * residuals / safe_dists  # (n_train,)
        grad = np.sum(scale[:, np.newaxis] * diff, axis=0)  # (q,)
        return grad

    best_result = None
    best_cost = float("inf")

    for restart in range(OOS_N_RESTARTS):
        # Initialize: random point near centroid of training embeddings
        centroid = z_train.mean(axis=0)
        z0 = centroid + rng.randn(q) * z_train.std()

        result = minimize(
            objective,
            z0,
            jac=gradient,
            method=OOS_OPTIM_METHOD,
            options={"maxiter": OOS_OPTIM_MAXITER, "ftol": 1e-12, "gtol": 1e-8},
        )

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    return best_result.x


def evaluate_generalization(
    D_full: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    q: int,
    seed: int,
) -> dict:
    """Run one fold of expert-level cross-validation.

    Parameters
    ----------
    D_full : np.ndarray
        Full n×n Oracle distance matrix.
    train_idx : np.ndarray
        Indices of training experts.
    test_idx : np.ndarray
        Indices of held-out experts.
    q : int
        Embedding dimensionality.
    seed : int
        Random seed.

    Returns
    -------
    dict with test→train and test→test metrics.
    """
    n_train = len(train_idx)
    n_test = len(test_idx)

    # Extract training sub-matrix
    D_train = D_full[np.ix_(train_idx, train_idx)]

    # Run SMACOF on training sub-matrix
    Z_train = run_smacof_train(D_train, q, seed)

    # ── Embed each test expert ────────────────────
    Z_test = np.zeros((n_test, q))
    for t, test_exp in enumerate(test_idx):
        # Distances from test expert to all training experts
        d_to_train = D_full[test_exp, train_idx]

        z_test = embed_single_test_expert(
            Z_train, d_to_train, q,
            seed=(seed * 10000 + t) % (2**32 - 1),
        )
        Z_test[t] = z_test

    # ── Evaluate Test→Train ───────────────────────
    # For each test expert, compare embedding distance to training experts
    # vs true Oracle distance
    test_train_embed_dists = []
    test_train_oracle_dists = []

    for t, test_exp in enumerate(test_idx):
        for tr, train_exp in enumerate(train_idx):
            embed_d = np.linalg.norm(Z_test[t] - Z_train[tr])
            oracle_d = D_full[test_exp, train_exp]
            test_train_embed_dists.append(embed_d)
            test_train_oracle_dists.append(oracle_d)

    test_train_embed_dists = np.array(test_train_embed_dists)
    test_train_oracle_dists = np.array(test_train_oracle_dists)

    tt_spearman = spearmanr(test_train_oracle_dists, test_train_embed_dists).statistic
    tt_pearson = pearsonr(test_train_oracle_dists, test_train_embed_dists).statistic
    tt_rmse = np.sqrt(np.mean((test_train_embed_dists - test_train_oracle_dists) ** 2))
    tt_mae = np.mean(np.abs(test_train_embed_dists - test_train_oracle_dists))

    # Normalized stress for test→train
    tt_stress_num = np.sum((test_train_embed_dists - test_train_oracle_dists) ** 2)
    tt_stress_den = np.sum(test_train_oracle_dists ** 2)
    tt_norm_stress = np.sqrt(tt_stress_num / max(tt_stress_den, 1e-15))

    # ── Evaluate Test→Test ────────────────────────
    # Compare embedding distances between test experts vs true Oracle distances
    test_test_embed_dists = []
    test_test_oracle_dists = []

    for i in range(n_test):
        for j in range(i + 1, n_test):
            embed_d = np.linalg.norm(Z_test[i] - Z_test[j])
            oracle_d = D_full[test_idx[i], test_idx[j]]
            test_test_embed_dists.append(embed_d)
            test_test_oracle_dists.append(oracle_d)

    test_test_embed_dists = np.array(test_test_embed_dists)
    test_test_oracle_dists = np.array(test_test_oracle_dists)

    # Handle edge cases with very few test pairs
    if len(test_test_embed_dists) < 3:
        ttt_spearman = float("nan")
        ttt_pearson = float("nan")
    else:
        ttt_spearman = spearmanr(test_test_oracle_dists, test_test_embed_dists).statistic
        ttt_pearson = pearsonr(test_test_oracle_dists, test_test_embed_dists).statistic

    ttt_rmse = np.sqrt(np.mean((test_test_embed_dists - test_test_oracle_dists) ** 2))
    ttt_mae = np.mean(np.abs(test_test_embed_dists - test_test_oracle_dists))

    ttt_stress_num = np.sum((test_test_embed_dists - test_test_oracle_dists) ** 2)
    ttt_stress_den = np.sum(test_test_oracle_dists ** 2)
    ttt_norm_stress = np.sqrt(ttt_stress_num / max(ttt_stress_den, 1e-15))

    return {
        "q": q,
        "n_train": n_train,
        "n_test": n_test,
        # Test→Train metrics
        "test_train_spearman": float(tt_spearman),
        "test_train_pearson": float(tt_pearson),
        "test_train_rmse": float(tt_rmse),
        "test_train_mae": float(tt_mae),
        "test_train_norm_stress": float(tt_norm_stress),
        # Test→Test metrics
        "test_test_spearman": float(ttt_spearman),
        "test_test_pearson": float(ttt_pearson),
        "test_test_rmse": float(ttt_rmse),
        "test_test_mae": float(ttt_mae),
        "test_test_norm_stress": float(ttt_norm_stress),
    }


def run_cv_for_matrix(
    D: np.ndarray,
    label: str,
    layer: str,
    realization_id: int = 0,
) -> list[dict]:
    """Run full 5-fold × 10-rep CV for a given distance matrix.

    Parameters
    ----------
    D : np.ndarray
        n×n symmetric distance matrix.
    label : str
        "oracle", "null_a", or "null_b".
    layer : str
        Layer name.
    realization_id : int
        Null realization index (0 for oracle).

    Returns
    -------
    list of dicts with per-fold results.
    """
    n = D.shape[0]
    all_results = []

    n_reps = N_REPETITIONS if label == "oracle" else 1
    for rep in range(n_reps):
        # Use deterministic seed per repetition
        rep_seed = RANDOM_SEED + rep * 7919  # large prime spacing
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep_seed)

        for fold, (train_idx, test_idx) in enumerate(kf.split(range(n))):
            for q in Q_VALUES:
                fold_seed = rep_seed * 100 + fold * 10 + q
                result = evaluate_generalization(D, train_idx, test_idx, q, fold_seed)
                result.update({
                    "layer": layer,
                    "label": label,
                    "realization": realization_id,
                    "repetition": rep,
                    "fold": fold,
                    "seed": fold_seed,
                })
                all_results.append(result)

    return all_results


def load_distance_matrix(layer: str) -> np.ndarray:
    """Load Oracle distance matrix for a layer."""
    path = os.path.join(RESULTS_DIR, f"oracle_distance_matrix_{layer}.csv")
    df = pd.read_csv(path)
    if df.iloc[:, 0].dtype == object:
        df = df.iloc[:, 1:]
    return df.values.astype(float)


def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 3B — PHASE 3: EXPERT-LEVEL CROSS-VALIDATION")
    print("=" * 70)

    oracle_results = []
    null_a_results = []
    null_b_results = []

    for layer in LAYERS:
        print(f"\n{'═' * 60}")
        print(f"LAYER: {layer}")
        print(f"{'═' * 60}")

        D_oracle = load_distance_matrix(layer)
        print(f"[Phase 3] Loaded Oracle distance matrix: {D_oracle.shape}")

        # ── Oracle CV ─────────────────────────────────
        print(f"\n[Phase 3] Running Oracle CV ({N_FOLDS} folds × {N_REPETITIONS} reps)...")
        layer_oracle = run_cv_for_matrix(D_oracle, "oracle", layer)
        oracle_results.extend(layer_oracle)
        print(f"  Completed {len(layer_oracle)} fold-q evaluations")

        # Quick summary for Oracle
        df_tmp = pd.DataFrame(layer_oracle)
        for q in Q_VALUES:
            q_data = df_tmp[df_tmp["q"] == q]
            spear_mean = q_data["test_test_spearman"].mean()
            spear_std = q_data["test_test_spearman"].std()
            print(f"  Oracle q={q:2d}: test→test ρ = {spear_mean:.4f} ± {spear_std:.4f}")

        # ── Null A CV ─────────────────────────────────
        print(f"\n[Phase 3] Running Null A CV ({N_NULL_REALIZATIONS} realizations)...")
        for r in range(N_NULL_REALIZATIONS):
            seed_a = RANDOM_SEED * 1000 + r * 100 + hash(layer) % 1000
            D_null_a = generate_null_a(D_oracle, seed=seed_a)
            layer_null_a = run_cv_for_matrix(D_null_a, "null_a", layer, realization_id=r)
            null_a_results.extend(layer_null_a)
            if (r + 1) % 10 == 0:
                print(f"  Completed {r + 1}/{N_NULL_REALIZATIONS} Null A realizations")

        # ── Null B CV ─────────────────────────────────
        print(f"\n[Phase 3] Running Null B CV ({N_NULL_REALIZATIONS} realizations)...")
        for r in range(N_NULL_REALIZATIONS):
            seed_b = RANDOM_SEED * 2000 + r * 100 + hash(layer) % 1000
            D_null_b = generate_null_b(seed=seed_b)
            layer_null_b = run_cv_for_matrix(D_null_b, "null_b", layer, realization_id=r)
            null_b_results.extend(layer_null_b)
            if (r + 1) % 10 == 0:
                print(f"  Completed {r + 1}/{N_NULL_REALIZATIONS} Null B realizations")

    # ── Save Results ──────────────────────────────
    print("\n[Phase 3] Saving results...")
    oracle_df = pd.DataFrame(oracle_results)
    null_a_df = pd.DataFrame(null_a_results)
    null_b_df = pd.DataFrame(null_b_results)

    save_csv(oracle_df, os.path.join(RESULTS_DIR, "oracle_cv_results.csv"))
    save_csv(null_a_df, os.path.join(RESULTS_DIR, "null_a_cv_results.csv"))
    save_csv(null_b_df, os.path.join(RESULTS_DIR, "null_b_cv_results.csv"))

    print(f"\n  Oracle CV results: {len(oracle_df)} rows")
    print(f"  Null A CV results: {len(null_a_df)} rows")
    print(f"  Null B CV results: {len(null_b_df)} rows")

    print("\n" + "=" * 70)
    print("PHASE 3 — EXPERT-LEVEL CROSS-VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
