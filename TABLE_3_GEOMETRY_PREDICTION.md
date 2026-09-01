# TABLE 3: GEOMETRY → FUNCTIONAL CONSEQUENCE

**Source Reference:** `results/exp4/final_report.md`

### Table 3A: Predictive Power Models

*Middle-Layer Only. N=5 independent cross-validation partitions (3 folds per partition).*
*Target: `Oracle_KL` capability drift.*

| Model Designation | Features Used | Mean Spearman $\rho$ | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **Model A** (Local) | 11 pre-merge standard features | +0.4797 | [+0.4464, +0.5050] |
| **Model B** (Geometry) | $L_2$ distance in $q=4$ MDS space | +0.7504 | [+0.7132, +0.7877] |
| **Model C** (CARE) | 11 Local + 1 Geometry (12 total) | +0.8146 | [+0.7848, +0.8430] |

*Note: Model A and C were trained via XGBoost (retrained per fold). Model B is an unlearned $L_2$ geometric distance.*

### Table 3B: Top-K Budget Performance (Precision@K)

*Highlights the performance inversion at highly selective thresholds.*

| Budget Selection | Model A (Local) | Model B (Geometry) | Model C (Local + Geometry) | Dominant Model |
| :--- | :--- | :--- | :--- | :--- |
| **Top 10 ($K=10$)** | 0.17 | **0.37** | 0.23 | **Model B** (Pure Geometry) |
| **Top 25 ($K=25$)** | 0.36 | **0.50** | 0.41 | **Model B** (Pure Geometry) |
| **Top 50 ($K=50$)** | 0.52 | 0.63 | **0.69** | **Model C** (Combined) |

**Key Observation:** In the highly selective $K=10$ and $K=25$ regimes, pure capability geometry (Model B) strongly dominates the combined model (Model C). Local descriptors (e.g., Usage Frequency, Output Similarity) actually degrade selective rank ordering, confirming their utility is highly budget-dependent.
