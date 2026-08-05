# CARE-MoE Experiment 1.5 — Linearization Gap Analysis

## Research Question

> Can the existing feature family, when combined, explain Oracle Capability Drift?

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Total pairs in dataset | 6,048 |
| Training pairs | 1,488 |
| Testing pairs | 1,488 |
| Discarded cross-boundary pairs | 3,072 |
| Number of base features | 7 |
| Active features (LASSO) | 5 |
| Dead features (LASSO) | 2 |
| Expert split | Train: 0–31, Test: 32–63 |
| Seq_Len filter | 256 |
| Layers | ['first', 'last', 'middle'] |
| Scaling | RobustScaler (fit on training set) |

---

## 2. Model Performance

| Model            | Variant   |   N_Features |   Spearman |   Pearson |    MAE |   RMSE |      R2 |
|:-----------------|:----------|-------------:|-----------:|----------:|-------:|-------:|--------:|
| XGBoost          | A         |            7 |     0.6568 |    0.4351 | 0.0015 | 0.0026 |  0.0229 |
| LinearRegression | A         |            7 |     0.5779 |    0.5964 | 0.0016 | 0.0022 |  0.3324 |
| Ridge            | A         |            7 |     0.5773 |    0.5963 | 0.0016 | 0.0022 |  0.3322 |
| LASSO            | A         |            7 |     0.5475 |    0.5975 | 0.0016 | 0.0022 |  0.3197 |
| XGBoost          | B         |            8 |     0.6774 |    0.4455 | 0.0014 | 0.0026 |  0.0383 |
| LASSO            | B         |            8 |     0.5475 |    0.5975 | 0.0016 | 0.0022 |  0.3197 |
| Ridge            | B         |            8 |     0.4523 |    0.5069 | 0.0018 | 0.0024 |  0.2111 |
| LinearRegression | B         |            8 |     0.4448 |    0.5001 | 0.0018 | 0.0024 |  0.1993 |
| XGBoost          | C         |           15 |     0.6755 |    0.4329 | 0.0014 | 0.0027 | -0.0182 |
| LASSO            | C         |           15 |     0.5459 |    0.6005 | 0.0016 | 0.0022 |  0.3233 |
| Ridge            | C         |           15 |     0.5354 |    0.5709 | 0.0016 | 0.0022 |  0.2850 |
| LinearRegression | C         |           15 |     0.5244 |    0.5531 | 0.0017 | 0.0023 |  0.2333 |

---

## 3. Linearization Gap

| Metric | Value |
|---|---|
| Best Linear Model | LinearRegression_A |
| Best Linear ρ | 0.5779 |
| Best Tree Model | XGBoost_B |
| Best Tree ρ | 0.6774 |
| **Δ_gap** | **+0.0995** |

### Interpretation

The linearization gap is **moderate** (Δ = 0.0995). There is some nonlinear structure the linear model cannot capture, but the gap is small enough that the existing features carry most of the signal. Experiment 2 may yield marginal improvements.

---

## 4. Feature Analysis

### 4.1 LASSO Coefficients (α = 0.0001)

| Feature | Coefficient |
|---|---|
| Usage_Frequency | +0.001566 |
| Jaccard_Overlap | +0.000342 |
| Routing_Similarity | -0.000339 |
| Weight_Cosine | -0.000181 |
| Weight_Distance | -0.000065 |
| Activation_Similarity | -0.000000 ⚠️ DEAD |
| Output_Similarity | +0.000000 ⚠️ DEAD |

**Dead features** (coefficient = 0, mathematically eliminated by LASSO):
Activation_Similarity, Output_Similarity

**Active features** (5):
Weight_Distance, Weight_Cosine, Routing_Similarity, Usage_Frequency, Jaccard_Overlap

### 4.2 Depth Effects

| Comparison | Δ Spearman |
|---|---|
| Model B (+ depth) vs Model A (global) | -0.0304 |
| Model C (+ interactions) vs Model B (+ depth) | -0.0016 |

Adding relative layer depth **improves** prediction, confirming layer-dependent behavior.

Feature × depth interactions provide **no meaningful** additional value.

---

## 5. Figures

### 5.1 Correlation Heatmap
![Correlation heatmap](./figures/01_correlation_heatmap.png)

### 5.2 LASSO Coefficients
![LASSO coefficients](./figures/02_lasso_coefficients.png)

### 5.3 XGBoost Feature Importance
![XGBoost feature importance](./figures/03_xgboost_importance.png)

### 5.4 Predicted vs Oracle Scatter
![Predicted vs Oracle scatter](./figures/04_predicted_vs_oracle.png)

### 5.5 Residual Plot
![Residual plot](./figures/05_residual_plot.png)

### 5.6 Linearization Gap Summary
![Linearization Gap summary](./figures/06_linearization_gap.png)

---

## 6. Scientific Conclusion

### Q1: Can the existing feature family explain Oracle Capability Drift?

**Partially.** The linear model achieves ρ = 0.5779, but the tree model reaches ρ = 0.6774, leaving a gap of 0.0995.

### Q2: Does a nonlinear model significantly outperform a linear model?

**Yes.** The gap of 0.0995 indicates nonlinear structure beyond linear feature combinations.

### Q3: Is Experiment 2 scientifically justified?

**Conditionally.** The gap (0.0995) suggests room for improvement but is not extreme.

---

## 7. Final Recommendation

**Outcome B (Marginal) — Experiment 2 is recommended but not critical.**

The existing feature family captures most of the signal, but a moderate linearization gap suggests some nonlinear interactions remain unexploited. Experiment 2's objective: discover new pairwise features that shrink the Linearization Gap and enable a simple linear model to approach the nonlinear ceiling.

---

*Report generated by CARE-MoE Experiment 1.5 pipeline.*
*Seed: 42 | Split: Strict Disjoint Expert (0–31 / 32–63)*
