"""
EXPERIMENT 7B - STATISTICAL UTILITIES
=====================================
Provides statistical tests, Spearman rank correlation, partial correlation,
bootstrap confidence intervals, and prediction calibration.
"""

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

def compute_spearman(x, y):
    """Computes Spearman rank correlation rho and two-tailed p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)

def compute_pearson(x, y):
    """Computes Pearson correlation r and two-tailed p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    res = stats.pearsonr(x, y)
    return float(res.statistic), float(res.pvalue)

def compute_partial_spearman(x, y, z):
    """
    Computes partial Spearman rank correlation rho(x, y | z),
    measuring the association between x and y after controlling for z.
    Calculated via Pearson correlation on rank residuals.
    """
    x = stats.rankdata(x)
    y = stats.rankdata(y)
    z = stats.rankdata(z)
    
    # Regress x on z, y on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    res_x = x - (slope_xz * z + intercept_xz)
    
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    res_y = y - (slope_yz * z + intercept_yz)
    
    return compute_pearson(res_x, res_y)

def bootstrap_correlation_ci(x, y, n_boot=10000, ci_level=0.95, seed=42):
    """
    Computes non-parametric bootstrap confidence interval for Spearman rho.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)
    
    boot_rhos = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sample_x = x[indices]
        sample_y = y[indices]
        if np.all(sample_x == sample_x[0]) or np.all(sample_y == sample_y[0]):
            continue
        rho, _ = stats.spearmanr(sample_x, sample_y)
        if not np.isnan(rho):
            boot_rhos.append(rho)
            
    if len(boot_rhos) == 0:
        return 0.0, 0.0
        
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_rhos, alpha * 100))
    ci_upper = float(np.percentile(boot_rhos, (1.0 - alpha) * 100))
    return ci_lower, ci_upper

def fit_monotonic_calibration(train_pred, train_actual):
    """
    Fits an Isotonic Regression mapping from predictions to actual values,
    guaranteeing monotonic calibration without changing rankings.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_pred, train_actual)
    return iso
