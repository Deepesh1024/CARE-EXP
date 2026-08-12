"""
CARE-MoE Experiment 4 — Metrics
=================================
All evaluation metrics per fold and aggregation across partitions.

Primary comparison: Δrho_CA = rho_C - rho_A (CARE vs local baseline)
Secondary:          Δrho_BA = rho_B - rho_A (geometry-only vs local)

Statistical units: 5 INDEPENDENT partitions.
  - 3 folds within a partition are CORRELATED.
  - Bootstrap CI computed over 5 partition-level effects.
  - N=5 is small; CI width is reported explicitly.
  - Wilcoxon signed-rank is a secondary descriptive test.

Statistical power note (pre-registered):
  With N=5 partition-level observations, bootstrap CI reflects
  sampling variability over 5 units. Results should NOT be
  interpreted as high-power inference. Stated explicitly in report.

Precision@K:
  K values = [10, 25, 50] absolute.
  "Safe" pairs = lowest Oracle KL values in test set.
  Pooled ranking across folds requires rank-normalization within fold.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DELTA_RHO_MIN,
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_CI_LEVEL,
    PRECISION_K_VALUES,
    RANDOM_SEED,
)


# ══════════════════════════════════════════════════════════
# Per-Fold Metrics
# ══════════════════════════════════════════════════════════

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_pred_c: np.ndarray,
    partition: int,
    fold: int,
) -> dict:
    """Compute all metrics for one fold across Models A, B, C.

    Model B is the geometry distance (NOT a learned predictor).
    All three models evaluated against the same y_true.

    Parameters
    ----------
    y_true   : (n_pairs,) Oracle KL targets.
    y_pred_a : (n_pairs,) Model A predictions.
    y_pred_b : (n_pairs,) Model B predictions (geometry distances).
    y_pred_c : (n_pairs,) Model C predictions.

    Returns
    -------
    dict with per-model metrics and Δrho values.
    """
    n = len(y_true)
    assert n >= 3, f"Too few test pairs ({n}) for correlation metrics."

    results = {
        "partition": partition,
        "fold": fold,
        "n_test_pairs": n,
    }

    for label, y_pred in [("A", y_pred_a), ("B", y_pred_b), ("C", y_pred_c)]:
        rho = float(spearmanr(y_true, y_pred).statistic)
        r = float(pearsonr(y_true, y_pred).statistic)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        results[f"rho_{label}"] = rho
        results[f"r_{label}"] = r
        results[f"rmse_{label}"] = rmse
        results[f"mae_{label}"] = mae
        results[f"r2_{label}"] = r2

    # Primary comparisons
    results["delta_rho_BA"] = results["rho_B"] - results["rho_A"]
    results["delta_rho_CA"] = results["rho_C"] - results["rho_A"]
    results["delta_rho_CB"] = results["rho_C"] - results["rho_B"]

    # Precision@K and Recall@K
    for k in PRECISION_K_VALUES:
        for label, y_pred in [("A", y_pred_a), ("B", y_pred_b), ("C", y_pred_c)]:
            prec, rec = _precision_recall_at_k(y_true, y_pred, k)
            results[f"prec_at_{k}_{label}"] = prec
            results[f"rec_at_{k}_{label}"] = rec

    return results


def _precision_recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Compute Precision@K and Recall@K.

    "Safe" pairs = lowest Oracle KL values.
    Predicted "safe" = pairs with lowest predicted values.

    If fewer than k pairs available, use all available pairs.
    """
    n = len(y_true)
    k_actual = min(k, n)

    # True positives: pairs with lowest Oracle KL (actual safe)
    true_safe_idx = set(np.argsort(y_true)[:k_actual])

    # Predicted positives: pairs with lowest predicted values
    pred_safe_idx = set(np.argsort(y_pred)[:k_actual])

    hits = len(true_safe_idx & pred_safe_idx)
    precision = hits / k_actual if k_actual > 0 else 0.0
    recall = hits / k_actual if k_actual > 0 else 0.0  # same denom since both top-k

    return float(precision), float(recall)


# ══════════════════════════════════════════════════════════
# Partition-Level Aggregation
# ══════════════════════════════════════════════════════════

def aggregate_partition(fold_results: list) -> dict:
    """Average fold-level metrics within one partition.

    Parameters
    ----------
    fold_results : list of 3 dicts from compute_fold_metrics.

    Returns
    -------
    dict with mean per-model metrics and Δrho for this partition.
    """
    assert len(fold_results) == 3, (
        "Expected 3 folds per partition for aggregation."
    )
    partition = fold_results[0]["partition"]
    agg = {"partition": partition, "n_folds": 3}

    keys = [
        "rho_A", "rho_B", "rho_C",
        "r_A", "r_B", "r_C",
        "rmse_A", "rmse_B", "rmse_C",
        "mae_A", "mae_B", "mae_C",
        "r2_A", "r2_B", "r2_C",
        "delta_rho_BA", "delta_rho_CA", "delta_rho_CB",
    ]
    for k in keys:
        vals = [f[k] for f in fold_results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
        agg[f"{k}_folds"] = vals

    # Precision@K partition means
    for k_val in PRECISION_K_VALUES:
        for model in ["A", "B", "C"]:
            pk = f"prec_at_{k_val}_{model}"
            rk = f"rec_at_{k_val}_{model}"
            agg[f"{pk}_mean"] = float(np.mean([f[pk] for f in fold_results]))
            agg[f"{rk}_mean"] = float(np.mean([f[rk] for f in fold_results]))

    return agg


# ══════════════════════════════════════════════════════════
# Statistical Inference Over 5 Partitions
# ══════════════════════════════════════════════════════════

def compute_final_statistics(partition_results: list) -> dict:
    """Compute statistics over the 5 independent partition-level effects.

    Statistical unit = partition (N=5).
    Bootstrap CI is computed for completeness; power is explicitly limited.

    Parameters
    ----------
    partition_results : list of 5 partition-level aggregation dicts.

    Returns
    -------
    dict with final statistics, CIs, Wilcoxon results, and decision gates.
    """
    assert len(partition_results) == 5, (
        f"Expected 5 partition-level results, got {len(partition_results)}."
    )

    rng = np.random.RandomState(RANDOM_SEED)

    def _boot_ci(vals: np.ndarray) -> tuple[float, float, float]:
        """Bootstrap 95% CI over 5 partition effects."""
        means = [np.mean(rng.choice(vals, len(vals), replace=True))
                 for _ in range(BOOTSTRAP_N_SAMPLES)]
        lo = float(np.percentile(means, 100 * (1 - BOOTSTRAP_CI_LEVEL) / 2))
        hi = float(np.percentile(means, 100 * (1 + BOOTSTRAP_CI_LEVEL) / 2))
        return float(np.mean(vals)), lo, hi

    stats = {
        "n_partitions": 5,
        "n_folds_per_partition": 3,
        "statistical_unit": "partition",
        "statistical_power_note": (
            "N=5 independent partition-level effects. Bootstrap CI reflects "
            "sampling variability over 5 units only. Results must NOT be "
            "interpreted as high-power inference."
        ),
        "bootstrap_n_samples": BOOTSTRAP_N_SAMPLES,
        "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
    }

    # Per-partition effects (listed explicitly)
    for key in ["delta_rho_BA_mean", "delta_rho_CA_mean",
                "rho_A_mean", "rho_B_mean", "rho_C_mean"]:
        vals = np.array([p[key] for p in partition_results])
        stats[f"partitions_{key}"] = vals.tolist()
        mean, lo, hi = _boot_ci(vals)
        stats[f"mean_{key}"] = mean
        stats[f"median_{key}"] = float(np.median(vals))
        stats[f"ci95_lo_{key}"] = lo
        stats[f"ci95_hi_{key}"] = hi

    # Wilcoxon signed-rank (secondary descriptive test, n=5)
    for delta_key in ["delta_rho_BA_mean", "delta_rho_CA_mean"]:
        vals = np.array([p[delta_key] for p in partition_results])
        try:
            # Wilcoxon test: are effects > 0?
            stat, pval = wilcoxon(vals, alternative="greater")
            stats[f"wilcoxon_stat_{delta_key}"] = float(stat)
            stats[f"wilcoxon_pval_{delta_key}"] = float(pval)
            stats[f"wilcoxon_note"] = (
                "Wilcoxon signed-rank test (n=5). Treat as secondary descriptive "
                "statistic only. Low power with N=5."
            )
        except Exception as e:
            stats[f"wilcoxon_stat_{delta_key}"] = None
            stats[f"wilcoxon_pval_{delta_key}"] = None
            stats[f"wilcoxon_error_{delta_key}"] = str(e)

    # Decision gates (pre-registered, evaluated post-hoc)
    delta_ba = stats["mean_delta_rho_BA_mean"]
    delta_ca = stats["mean_delta_rho_CA_mean"]
    ci_ba_lo = stats["ci95_lo_delta_rho_BA_mean"]
    ci_ca_lo = stats["ci95_lo_delta_rho_CA_mean"]

    stats["decision"] = _evaluate_decision_gates(
        delta_ba, delta_ca, ci_ba_lo, ci_ca_lo
    )

    return stats


def _evaluate_decision_gates(
    delta_ba: float,
    delta_ca: float,
    ci_ba_lo: float,
    ci_ca_lo: float,
) -> dict:
    """Evaluate pre-registered decision gates.

    CASE A: Geometry fails.         delta_ba < 0.05 OR CI includes 0.
    CASE B: Geometry adds value.    delta_ca >= 0.05 AND CI excludes 0. (H10 survives)
    CASE C: Geometry dominates.     delta_ba >= 0.05 AND CI excludes 0.
    CASE D: Geometry complementary. B fails but C succeeds.
    CASE E: Geometry subsumes local. B succeeds, C no meaningful gain over B.
    """
    threshold = DELTA_RHO_MIN
    ci_ba_excludes_zero = ci_ba_lo > 0
    ci_ca_excludes_zero = ci_ca_lo > 0

    cases = {}

    # CASE A
    cases["A_geometry_fails"] = (delta_ba < threshold) or (not ci_ba_excludes_zero)

    # CASE B: H10 survives
    cases["B_geometry_adds_value_H10_survives"] = (
        delta_ca >= threshold and ci_ca_excludes_zero
    )

    # CASE C
    cases["C_geometry_dominates"] = (
        delta_ba >= threshold and ci_ba_excludes_zero
    )

    # CASE D
    cases["D_geometry_complementary"] = (
        cases["A_geometry_fails"] and cases["B_geometry_adds_value_H10_survives"]
    )

    # CASE E (approximate)
    cases["E_geometry_subsumes_local"] = (
        cases["C_geometry_dominates"] and (delta_ca - delta_ba) < threshold
    )

    # H10 verdict
    cases["H10_survives"] = cases["B_geometry_adds_value_H10_survives"]

    cases["delta_rho_min_threshold"] = threshold
    cases["delta_ba"] = delta_ba
    cases["delta_ca"] = delta_ca
    cases["ci_ba_lo"] = ci_ba_lo
    cases["ci_ca_lo"] = ci_ca_lo

    return cases
