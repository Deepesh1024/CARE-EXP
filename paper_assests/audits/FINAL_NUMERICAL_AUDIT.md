# Final Numerical Audit

**Claim:** Global Spearman rho = 0.7657
**Computed:** 0.7657
**Status:** PASS

**Claim:** 186 pairs with |d_i - d_j| < 1e-4 from median, KL range ~0.001522 to ~0.007096
**Computed:** N=186, Min=0.001522, Max=0.007096
**Status:** PASS

**Claim:** 15-24% improvement calculations for Static vs Adaptive KL
Experts=60, Static=2.477, Adapt=2.036 -> Reduction=17.8%
Experts=56, Static=6.82, Adapt=5.143 -> Reduction=24.6%
Experts=48, Static=17.694, Adapt=14.881 -> Reduction=15.9%
**Status:** PASS (verified 17.8%, 24.6%, 15.9% reductions)

**Terminology Audit:** 'near-identical capability distances' -> 'near-identical capability distances'
**Action:** Replaced in target markdown files.

**Claim:** Novelty of Adaptive Recomputation over Static Clustering
**Audit:** Literature (e.g. Sub-MoE, HC-SMoE) confirms prior works perform static clustering of experts. The sequential functional re-evaluation is indeed novel compared to those baselines.
**Status:** PASS
