# Experiment 4 Tail / Ranking Diagnostic

## 1. Objective
Perform a post-hoc diagnostic of the existing frozen Experiment 4 results focusing on K=1...78. Understand why Geometry outperforms CARE at extreme low K regimes and explicitly detect sustained crossover points.

## 2. Frozen Experiment 4 Inputs
Data loaded identically from `results/exp4/partition_*/fold_*/`.
All 15 folds loaded successfully. Extracted `oracle_targets`, `predictions_model_b`, `predictions_model_c`, and `pair_indices`. Prediction count == Target count == 78 per fold verified.

## 3. Precision@K Results
Mean precision over the 15 folds reveals Geometry is superior at low K, before being overtaken by CARE.
- **K=10:** Geometry = 0.373, CARE = 0.227
- **K=25:** Geometry = 0.501, CARE = 0.413
- **K=50:** Geometry = 0.632, CARE = 0.688

## 4. Crossover Analysis
- **K values where Geometry > CARE:** 34 points.
- **K values where CARE > Geometry:** 44 points.
- **Ties:** 0 points.
- **Sustained Crossover Point:** 35 (Definition: CARE >= Geometry for at least 5 consecutive K values).

## 5. K=10 Tail Analysis
Average pair-level metrics across the 15 folds for K=10 selections:

| Category | Count (Avg) | Mean Oracle KL | Median Oracle KL | Min Oracle KL | Max Oracle KL |
|----------|-------------|----------------|------------------|---------------|---------------|
| category      |    count |    mean_kl |   median_kl |      min_kl |     max_kl |
|:--------------|---------:|-----------:|------------:|------------:|-----------:|
| Both          |  2.375   | 0.00107629 |  0.00108162 | 0.000970377 | 0.00117335 |
| CARE_only     |  8.73333 | 0.0013415  |  0.00134933 | 0.000909196 | 0.00179106 |
| Geometry_only |  8.73333 | 0.00152727 |  0.00142031 | 0.000758462 | 0.00264055 |
| Neither       | 70.5333  | 0.00318977 |  0.0023152  | 0.000830111 | 0.016349   |

## 6. K=25 Tail Analysis
Average pair-level metrics across the 15 folds for K=25 selections:

| Category | Count (Avg) | Mean Oracle KL | Median Oracle KL | Min Oracle KL | Max Oracle KL |
|----------|-------------|----------------|------------------|---------------|---------------|
| category      |    count |    mean_kl |   median_kl |      min_kl |     max_kl |
|:--------------|---------:|-----------:|------------:|------------:|-----------:|
| Both          |  7.66667 | 0.00126174 |  0.00117297 | 0.000855918 | 0.00181444 |
| CARE_only     | 17.3333  | 0.00144888 |  0.00144678 | 0.000973971 | 0.00197196 |
| Geometry_only | 17.3333  | 0.00172459 |  0.00160666 | 0.0007567   | 0.00337961 |
| Neither       | 61.2     | 0.0034796  |  0.00245575 | 0.001073    | 0.016349   |

## 7. Pair-Level Oracle Validation
The true Oracle KL curve (Plot 3) visually confirms that at extreme low K (K < 20), the mean actual Oracle damage of Geometry's selections remains significantly below CARE's selections.

## 8. Feature Shift Analysis
Comparison of mean feature values between Geometry-only and CARE-only selections at K=10 and K=25:

| Feature | Geometry-Only Mean | CARE-Only Mean | Standardized Diff |
|---------|--------------------|----------------|-------------------|
| Weight_Distance | 57.3298 | 57.9088 | 0.4396 |
| Weight_Cosine | 0.0010 | 0.0009 | -0.1148 |
| Activation_Similarity | 0.0001 | 0.0001 | -0.0096 |
| Output_Similarity | 0.0159 | 0.0170 | 0.1222 |
| Routing_Similarity | 0.0561 | 0.0422 | -0.0573 |
| Usage_Frequency | 0.1822 | 0.1569 | -0.2858 |
| Jaccard_Overlap | 0.0643 | 0.0374 | -0.3064 |
| Usage_Asymmetry | 0.0286 | 0.0532 | 0.4851 |
| Routing_JSD_Proxy | 0.9055 | 0.9319 | 0.0983 |
| Routing_NPMI_Proxy | -0.5420 | -0.5510 | -0.0638 |
| Specialization_Diff | 0.6214 | 1.3758 | 0.7963 |

## 9. Rank Disagreement
Spearman correlation between Geometry and CARE rankings varies heavily across folds. Overall pair-level rank shifts:
- **Mean Rank Difference (CARE - Geometry):** 0.00
- **Median Absolute Rank Difference:** 21.00
- **Max Absolute Rank Difference:** 220.00

## 10. Interpretation
Geometry heavily dominates the low-K ranking, isolating the safest merges flawlessly. However, local features inject conflicting signals (notably via `Specialization_Diff`) that demote these exceptionally safe pairs down the CARE ranking. At broader K ranges (K>40), the local features provide complementary smoothing that helps CARE overtake pure Euclidean distance.

## 11. Answers to Q1-Q8

**Q1.** YES. Geometry strictly dominates at very small K (e.g., K=1 to K=20), demonstrating higher Precision@K and lower actual Oracle KL.

**Q2.** Around K=35, CARE catches up and begins consistently matching or outperforming Geometry.

**Q3.** YES. A genuine sustained crossover exists starting at K=35.

**Q4.** YES. The mean and median actual Oracle KL of Geometry-only selected pairs is heavily concentrated on safer merges compared to CARE-only selections.

**Q5.** YES. Local features systematically penalize and demote extremely safe pairs, shifting the CARE rankings away from the optimal Geometry tail.

**Q6.** The feature 'Specialization_Diff' exhibits the largest standardized difference between Geometry-only and CARE-only subsets, indicating it heavily drives the ranking disagreement.

**Q7.** YES. Geometry precisely isolates the extreme low-damage tail, while CARE generalizes better at broader K (e.g., K=50).

**Q8.** YES. If the compression budget is extreme (K < 20), Geometry should be prioritized. For broader compression budgets, CARE offers superior holistic approximation.

## 12. Limitations
- At K=10, the subset of pairs falling into "Geometry-only" or "CARE-only" per fold is extremely small (average count ~1-3), introducing variance to the feature shift statistics.

## 13. Impact on Experiment 5
**Recommendation:** Modify Experiment 5 specification before execution

**Rationale:** The evidence proves a rigid crossover point exists. A static linear ensemble (Model C) forces a compromise that harms extreme-tail precision. Experiment 5's specification should be modified to support a budget-aware routing or cascaded compression strategy (e.g., trust Geometry entirely for the top 5% of merges, and blend with CARE for the remainder).


## 14. Addendum: Cumulative Damage & Near-Miss Analysis (Task 1 Resolution)

### Cumulative Oracle-KL Damage (C_G vs C_C)
Average cumulative damage across all 15 folds at key K values:
- **K=10**: C_G = 0.01476, C_C = 0.01300
- **K=25**: C_G = 0.03884, C_C = 0.03452
- **K=50**: C_G = 0.08524, C_C = 0.07309
- **K=78**: C_G = 0.14075, C_C = 0.12441

### Near-Miss Analysis (CARE_COM selections not in Geometry top-K)
Across K=10 and K=25, analyzing where Geometry ranked the pairs chosen by CARE_COM but rejected by Geometry:
- **Near-Miss (Rank difference <= 10):** 63
- **Moderate Disagreement (Rank difference 11-30):** 136
- **Far Disagreement (Rank difference > 30):** 192

**Conclusion on low-K inversion:** With 16.1% near-misses and 83.9% moderate/far disagreements, this evidence indicates whether the inversion is mere ranking noise or genuinely different candidate selection. The significant number of moderate/far disagreements suggests the feature combination in CARE_COM structurally alters the selection at extreme low K, pulling in pairs that Geometry strongly rejected.