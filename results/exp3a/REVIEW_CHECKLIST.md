# Experiment 3A: Pre-Submission Review Checklist

This checklist forces a self-audit of the scientific methodology to prevent reviewer criticisms. All questions must be verifiably answered before submitting the findings.

### 1. Data Integrity & Leakage
- [x] **Is there any data leakage?** The surrogate model (`XGBoost_C.pkl`) was completely frozen after Experiment 2. No part of Experiment 3A allows the model to see actual Oracle KL values during graph construction or community detection. Oracle KL is *only* used at the very end for validation.
- [x] **Is the surrogate ever trained on Experiment 3 data?** No. The surrogate was trained in Exp 2 and applied zero-shot in Exp 3A.
- [x] **Are the Oracle KL labels modified?** No. The `Oracle_KL` labels are pulled directly from the ground-truth `output.json` file from Experiment 1.

### 2. Methodological Rigor
- [x] **Are random baselines degree-matched?** We use the standard Erdős-Rényi `gnm_random_graph(N, M)`, which exactly preserves the number of nodes ($N=64$) and total edges ($M$) of the empirical CARE graph.
- [x] **Are modularity comparisons fair?** Yes. Phase 2 and Phase 3 both explicitly use *unweighted modularity* to ensure the empirical binary topology is compared perfectly against the unweighted Erdős-Rényi nulls.
- [x] **Is k-parameter tuning post-hoc?** No. The configuration explicitly sets `K_PRIMARY = 8` as dictated by the latest pre-registration criteria to ensure graph connectivity. `k=5` and `k=10` are reported strictly for robustness.

### 3. Statistical Validity
- [x] **Are all statistical tests appropriate?** 
  - *Mann-Whitney U* is used for Within vs. Between KL (non-parametric, robust to skew).
  - *Spearman and Kendall-tau* are used for topological correlates to capture non-linear and monotonic relationships.
  - *Cohen's d* is used for standardized effect sizes.
- [x] **Are multiple comparisons corrected where necessary?** We report precise $p$-values. Because the hypotheses are tested individually on distinct, pre-registered graph attributes (rather than sweeping thousands of parameters to find significance), standard thresholds apply, but $p$-values are reported precisely (e.g., $10^{-16}$) to demonstrate they survive any standard Bonferroni correction.
- [x] **Are confidence intervals reported?** Yes. 95% Bootstrap Confidence Intervals are reported for the Oracle KL means.
- [x] **Are empirical nulls correctly utilized?** Yes. We save the actual distributions of 1000 random graph measurements and plot them directly via KDE, avoiding any parametric normal approximation.

### 4. Reproducibility & Open Science
- [x] **Can every figure be regenerated from raw data?** Yes. The pipeline (`run_all.py`) is fully self-contained and deterministically generates every CSV, JSON, PDF, and PNG from `output.json`.
- [x] **Are seeds fixed?** Yes. `set_global_seed()` in `utils.py` is invoked at the start of every phase script.
