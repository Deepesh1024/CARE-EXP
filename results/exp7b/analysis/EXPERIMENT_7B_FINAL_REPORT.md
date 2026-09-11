# Experiment 7B: Final Diagnostic Report
## CARE-COM Merge Failure Analysis and Joint Functional Interaction

**Generated:** 2026-09-11T16:49:11.129743  
**Sample Size:** N = 18 pairs (Stratified across Group A, B, C)  
**Decision Verdict:** **[KILL]**

---

### Pre-Registered Decision Gates

| Decision Gate | Metric / Target | Observed Value | 95% Bootstrap CI | Status |
|---|---|---|---|---|
| **GATE 1 (Baseline Predictive Utility)** | Spearman rho >= 0.30, p < 0.05 | rho = +0.6883 (p = 1.5871e-03) | [+0.2284, +0.9353] | **PASS** |
| **GATE 2 (Interaction vs Prediction Error)** | Spearman rho >= 0.40, CI > 0 | rho = -0.1909 (p = 4.4793e-01) | [-0.6184, +0.3440] | **FAIL** |

---

### Primary Hypothesis Outcomes
- **H0:** Joint functional interaction does not meaningfully explain CARE-COM merge errors.
- **H1:** Pairs with stronger non-additive joint functional interaction exhibit larger CARE-COM merge prediction errors.
- **Conclusion:** **H0 Retained (Falsification Confirmed)**.

---

### Key Figures Generated
1. `plots/scatter_pred_vs_actual.png`: Predicted vs actual merge damage.
2. `plots/scatter_interaction_vs_error.png`: Non-additive interaction vs CARE-COM prediction error.
3. `plots/degeneracy_classification.png`: 2D exploratory interaction landscape.
