# Experiment 3B: Capability Geometry Validation — Phase A

## Final Report

---

## 1. Scientific Question

**Hypothesis**: "The functional behavior of MoE experts may possess a lower-dimensional geometric structure that generalizes to unseen experts."

Phase A tests whether the ground-truth functional distances among experts contain statistically meaningful low-dimensional structure that generalizes to held-out experts, compared to null models.

## 2. Data Provenance

- **Source**: `/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp1/output.json`
- **Ground truth**: Oracle_KL (KL divergence from original to merged-expert model)
- **Seq_Len filter**: 512
- **Experts**: 64
- **Layers**: first, middle, last
- **Pairs per layer**: C(64,2) = 2016
- **XGBoost surrogate**: EXCLUDED from geometry (used only in Exp 2/3A)
- **Raw distributions**: NOT available (only scalar Oracle_KL)

### Distance Construction

**Method**: d(i,j) = Oracle_KL(i,j)  [inherently symmetric]

**Justification**: Oracle_KL measures KL(P_original || P_merged) where the merged model is created by UniformAverage of experts i and j. Since merge(i,j) = merge(j,i), Oracle_KL is inherently symmetric. No symmetrization formula needed.

**Layer first**: min=0.001544, max=0.025590, mean=0.005062, median=0.004343, triangle violations=75
**Layer middle**: min=0.000579, max=0.043282, mean=0.003602, median=0.002374, triangle violations=1166
**Layer last**: min=0.000547, max=0.016362, mean=0.005248, median=0.004978, triangle violations=943

## 3. Methodology

### Embedding Method

- **Algorithm**: Non-metric MDS (SMACOF)
- **metric**: True (non-metric)
- **max_iter**: 300
- **n_init**: 2
- **eps**: 0.0001
- **Dimensions tested**: q ∈ {2, 4, 6, 8}

### Cross-Validation

- **Expert-level holdout**: 5-fold × 2 repetitions = 10 total folds
- **Training experts**: ~52 per fold
- **Held-out experts**: ~12 per fold
- **Out-of-sample embedding**: Coordinate optimization with frozen Z_train
  - Method: L-BFGS-B
  - Restarts: 2
  - Objective: argmin_z Σ_i (||z - z_i||₂ - d_ji)²

### Null Models

- **Null A (Pairwise-Shuffled)**: 1 realizations
  - Upper-triangle distances randomly permuted
  - Preserves marginal distance distribution, destroys expert-identity structure
- **Null B (Random Euclidean)**: 1 realizations
  - 64 random points in R^64
  - Independent generic high-dimensional geometry baseline

## 4. Results

### Layer: first

| q | Oracle ρ | Null A ρ | Null B ρ | Oracle RMSE | Null A RMSE | Null B RMSE | Oracle Stress | Null A Stress | Null B Stress |
|---|---------|---------|---------|-------------|-------------|-------------|---------------|---------------|---------------|
| 2 | 0.6241 [0.5284, 0.7198] | -0.0081 [nan, nan] | 0.2634 [nan, nan] | 0.0019 [0.0017, 0.0020] | 0.0036 [nan, nan] | 4.9514 [nan, nan] | 0.3416 [0.3043, 0.3788] | 0.6245 [nan, nan] | 0.4377 [nan, nan] |
| 4 | 0.6762 [0.5489, 0.8035] | 0.0067 [nan, nan] | 0.2755 [nan, nan] | 0.0015 [0.0013, 0.0017] | 0.0032 [nan, nan] | 3.5203 [nan, nan] | 0.2736 [0.2219, 0.3253] | 0.5604 [nan, nan] | 0.3120 [nan, nan] |
| 6 | 0.5735 [0.4583, 0.6888] | -0.0434 [nan, nan] | 0.3488 [nan, nan] | 0.0016 [0.0013, 0.0020] | 0.0031 [nan, nan] | 2.8708 [nan, nan] | 0.2891 [0.2517, 0.3265] | 0.5411 [nan, nan] | 0.2544 [nan, nan] |
| 8 | 0.6117 [0.5237, 0.6997] | 0.0468 [nan, nan] | 0.3743 [nan, nan] | 0.0017 [0.0014, 0.0019] | 0.0030 [nan, nan] | 2.4309 [nan, nan] | 0.3060 [0.2539, 0.3581] | 0.5250 [nan, nan] | 0.2156 [nan, nan] |

### Layer: middle

| q | Oracle ρ | Null A ρ | Null B ρ | Oracle RMSE | Null A RMSE | Null B RMSE | Oracle Stress | Null A Stress | Null B Stress |
|---|---------|---------|---------|-------------|-------------|-------------|---------------|---------------|---------------|
| 2 | 0.7668 [0.6974, 0.8362] | 0.0061 [nan, nan] | 0.2812 [nan, nan] | 0.0011 [0.0009, 0.0014] | 0.0061 [nan, nan] | 4.8881 [nan, nan] | 0.3047 [0.2387, 0.3707] | 0.8887 [nan, nan] | 0.4333 [nan, nan] |
| 4 | 0.7805 [0.7099, 0.8512] | -0.0147 [nan, nan] | 0.3314 [nan, nan] | 0.0010 [0.0008, 0.0013] | 0.0059 [nan, nan] | 3.3896 [nan, nan] | 0.2724 [0.2151, 0.3297] | 0.8598 [nan, nan] | 0.3010 [nan, nan] |
| 6 | 0.7261 [0.6137, 0.8386] | -0.0252 [nan, nan] | 0.3861 [nan, nan] | 0.0010 [0.0008, 0.0013] | 0.0059 [nan, nan] | 2.6865 [nan, nan] | 0.2724 [0.2092, 0.3356] | 0.8646 [nan, nan] | 0.2392 [nan, nan] |
| 8 | 0.7195 [0.6572, 0.7817] | -0.0143 [nan, nan] | 0.3541 [nan, nan] | 0.0011 [0.0008, 0.0013] | 0.0059 [nan, nan] | 3.0024 [nan, nan] | 0.2832 [0.2261, 0.3403] | 0.8619 [nan, nan] | 0.2666 [nan, nan] |

### Layer: last

| q | Oracle ρ | Null A ρ | Null B ρ | Oracle RMSE | Null A RMSE | Null B RMSE | Oracle Stress | Null A Stress | Null B Stress |
|---|---------|---------|---------|-------------|-------------|-------------|---------------|---------------|---------------|
| 2 | 0.7788 [0.7297, 0.8279] | -0.0461 [nan, nan] | 0.3154 [nan, nan] | 0.0019 [0.0017, 0.0021] | 0.0033 [nan, nan] | 4.7633 [nan, nan] | 0.3233 [0.2984, 0.3482] | 0.6026 [nan, nan] | 0.4205 [nan, nan] |
| 4 | 0.7577 [0.6733, 0.8422] | -0.0423 [nan, nan] | 0.3443 [nan, nan] | 0.0019 [0.0014, 0.0023] | 0.0029 [nan, nan] | 3.5720 [nan, nan] | 0.3144 [0.2642, 0.3645] | 0.5268 [nan, nan] | 0.3156 [nan, nan] |
| 6 | 0.7387 [0.6842, 0.7931] | -0.0291 [nan, nan] | 0.3715 [nan, nan] | 0.0020 [0.0017, 0.0023] | 0.0028 [nan, nan] | 2.7625 [nan, nan] | 0.3364 [0.3020, 0.3708] | 0.5014 [nan, nan] | 0.2435 [nan, nan] |
| 8 | 0.6829 [0.5938, 0.7719] | 0.0044 [nan, nan] | 0.4138 [nan, nan] | 0.0021 [0.0017, 0.0025] | 0.0026 [nan, nan] | 2.4503 [nan, nan] | 0.3520 [0.3009, 0.4031] | 0.4783 [nan, nan] | 0.2167 [nan, nan] |

## 5. Statistical Comparisons

| Layer | q | Δρ(Oracle−NullA) | CI excl 0? | Δρ(Oracle−NullB) | CI excl 0? |
|-------|---|------------------|------------|------------------|------------|
| first | 2 | +0.6323 | No | +0.3608 | No |
| first | 4 | +0.6695 | No | +0.4007 | No |
| first | 6 | +0.6169 | No | +0.2248 | No |
| first | 8 | +0.5649 | No | +0.2374 | No |
| middle | 2 | +0.7607 | No | +0.4856 | No |
| middle | 4 | +0.7953 | No | +0.4491 | No |
| middle | 6 | +0.7513 | No | +0.3400 | No |
| middle | 8 | +0.7338 | No | +0.3654 | No |
| last | 2 | +0.8248 | No | +0.4634 | No |
| last | 4 | +0.8000 | No | +0.4134 | No |
| last | 6 | +0.7678 | No | +0.3672 | No |
| last | 8 | +0.6785 | No | +0.2691 | No |

## 6. Figures

### Primary Figure 1: Fidelity Curve (Test→Test Spearman ρ)

![Fidelity curve — Layer first](/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp3b/figures/fidelity_curve_first.png)

![Fidelity curve — Layer middle](/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp3b/figures/fidelity_curve_middle.png)

![Fidelity curve — Layer last](/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp3b/figures/fidelity_curve_last.png)

### Primary Figure 2: Stress Curve (Test→Test RMSE)

![Stress curve — Layer first](/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp3b/figures/stress_curve_first.png)

![Stress curve — Layer middle](/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp3b/figures/stress_curve_middle.png)

![Stress curve — Layer last](/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp3b/figures/stress_curve_last.png)

## 7. Data Leakage Audit

| Check | Status | Detail |
|-------|--------|--------|
| Ground-truth Oracle data used | ✅ **PASS** | Ground truth field: Oracle_KL, Source: /Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/result |
| XGBoost predictions excluded from geometry | ✅ **PASS** | XGBoost Predicted_KL from Experiment 2 is an approximation of Oracle KL. Using it as ground truth wo |
| Test experts excluded from Z_train fitting | ✅ **PASS** | run_smacof_train receives D_train (train×train submatrix only). Test expert indices are never includ |
| Z_train frozen during test embedding | ✅ **PASS** | embed_single_test_expert takes z_train as read-only input. Only the test coordinate z_j is optimized |
| Test-test distances excluded from test embedding | ✅ **PASS** | embed_single_test_expert receives only d_test_to_train (distances to training experts). Test-test di |
| q selected without using test performance | ✅ **PASS** | All q values are evaluated and reported. No single q is selected as 'optimal' using test performance |
| Oracle and null processed identically | ✅ **PASS** | run_cv_for_matrix is called identically for Oracle, Null A, and Null B. Same fold splits (matched se |

**Overall**: ALL PASS — EXPERIMENT VALID

## 8. Scientific Classification

### Classification: **B. PARTIAL SUPPORT**

Some low-dimensional structure is present in the Oracle geometry, but it is weak, unstable across layers, or requires relatively high dimensionality to emerge. Further investigation may be warranted but strong claims are not supported.

### Evidence Summary

- Total (layer, q) comparisons: 12
- Significant vs Null A: 0 (0.0%)
- Significant vs Null B: 0 (0.0%)
- Mean Oracle ρ: 0.7031
- Mean Null A ρ: -0.0133
- Mean Null B ρ: 0.3383
- Mean advantage vs Null A: +0.7163
- Mean advantage vs Null B: +0.3647

## 9. Important Distinctions

This report distinguishes three separate claims:

1. **Metric structure**: Oracle_KL defines a symmetric, non-negative function on expert pairs. Triangle inequality violations are documented above.
2. **Low-dimensional structure**: SMACOF stress curves indicate whether pairwise distances can be represented in fewer dimensions than the ambient space.
3. **Out-of-sample generalization**: Expert-level holdout tests whether the geometric structure extends to experts not used in embedding.

Phase A does **NOT** claim that a capability manifold exists. Successful MDS embedding is necessary but not sufficient evidence for manifold structure.

## 10. Software & Configuration

- Python: 3.13.9
- scikit-learn: 1.9.0
- scipy: 1.17.1
- numpy: 2.5.1
- pandas: 3.0.5
- matplotlib: 3.11.1
- Platform: macOS-26.6.1-arm64-arm-64bit-Mach-O
- Random seed: 42
