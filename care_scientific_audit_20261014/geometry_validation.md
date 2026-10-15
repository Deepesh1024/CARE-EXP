# Task 3 Audit Report: Geometry Prediction Experiment and Cross-Validation Integrity

**Audit Date:** October 14, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifacts Analyzed:**  
- Experiment 4 Code: [`experiments/experiment4/`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4/) (`cv_splits.py`, `data_loader.py`, `mds_embedding.py`, `model_a.py`, `model_b.py`, `model_c.py`, `metrics.py`, `report.py`)  
- Experiment 4 Records: [`results/exp4/fold_metrics.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/fold_metrics.csv), [`results/exp4/partition_metrics.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/partition_metrics.csv), [`results/exp4/final_report.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/final_report.json)  
- Published Assets: [`paper_assests/tables/table1_geometry.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/paper_assests/tables/table1_geometry.md), [`audit/CLAIM_AUDIT.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/CLAIM_AUDIT.md), [`audit/experiment4_pair_level.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/experiment4_pair_level.csv)

---

## 1. Executive Summary & Verification Matrix

The central claim audited in Task 3 is:
> *"Capability geometry predicts merge-induced functional damage."* (Claim 1)

Our forensic re-computation validates the mathematical arithmetic of the reported statistics, but uncovers **two critical methodological caveats** regarding transductive Oracle dependence and dyadic clustering.

### Exact Numerical Re-computation
| Metric | Reported Value | Recomputed Value | Mathematical Match? | Source Record |
|---|:---:|:---:|:---:|---|
| **Local Baseline Held-Out Spearman ($\rho_A$)** | **0.4797** | **0.479718** | **EXACT (100%)** | `partition_metrics.csv:rho_A_mean` |
| **Geometry-Only Held-Out Spearman ($\rho_B$)** | **0.7504** | **0.750449** | **EXACT (100%)** | `partition_metrics.csv:rho_B_mean` |
| **Combined Predictor Held-Out Spearman ($\rho_C$)** | **0.8146** | **0.814581** | **EXACT (100%)** | `partition_metrics.csv:rho_C_mean` |
| **Overall Pair-Level Spearman ($\rho$)** | **0.7657** | **0.765738** | **EXACT (100%)** | `audit/experiment4_pair_level.csv` |
| **Precision@10 (A / B / C)** | **0.17 / 0.37 / 0.23** | **0.1667 / 0.3733 / 0.2267** | **EXACT (Rounded)** | `partition_metrics.csv:prec_at_10_*` |
| **Precision@25 (A / B / C)** | **0.36 / 0.50 / 0.41** | **0.3600 / 0.5013 / 0.4133** | **EXACT (Rounded)** | `partition_metrics.csv:prec_at_25_*` |
| **Precision@50 (A / B / C)** | **0.52 / 0.63 / 0.69** | **0.5240 / 0.6320 / 0.6880** | **EXACT (Rounded)** | `partition_metrics.csv:prec_at_50_*` |

---

## 2. Cross-Validation Protocol & Leakage Audit

### A. Folds and Expert Disjointness
- **Design:** 5 independent partitions $\times$ 3 folds = **15 total evaluation folds**.
- **Unit of Generalization:** The **individual expert**, not the pair.
- **Rule Verification:**
  - In each partition, the 64 experts of `allenai/OLMoE-1B-7B-0924` (middle layer) are partitioned into 3 disjoint subsets of size 21, 21, and 22.
  - Training pairs $(i, j)$ require **both** $i, j \in \text{train\_experts}$.
  - Test evaluation pairs $(i, j)$ require **both** $i, j \in \text{test\_experts}$.
  - Cross pairs ($i \in \text{train}, j \in \text{test}$) are explicitly discarded.
- **Leakage Check Result:** **PASS.** No test-test pairs appear in the training feature matrix $X_{\text{train}}$ or target vector $y_{\text{train}}$. Preprocessing (`RobustScaler`) is fitted strictly on $X_{\text{train}}$ and transformed on $X_{\text{test}}$.

---

## 3. Critical Methodological Finding: The Transductive Oracle Embedding

While strict pair-level leakage rules were enforced, the audit reveals a fundamental conceptual limitation in what "Capability Geometry" actually represents in Experiment 4:

### How Model B Produces "Geometry"
In [`experiments/experiment4/mds_embedding.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4/mds_embedding.py):
1. **Training MDS:** Metric SMACOF embeds the training submatrix $D_{\text{train}} \in \mathbb{R}^{43 \times 43}$ into $Z_{\text{train}} \in \mathbb{R}^{43 \times 4}$.
2. **Out-of-Sample (OOS) Test Embedding:** For each held-out test expert $i \in \text{test}$, its coordinate $z_i \in \mathbb{R}^4$ is found via L-BFGS-B optimization:
   $$z_i = \arg\min_z \sum_{j \in \text{train}} \left( \|z - Z_j\|_2 - D_{\text{oracle}}(i, j) \right)^2$$
3. **Prediction:** For test pair $(i, j)$, the prediction is $\|z_i - z_j\|_2$.

### The Scientific Implication:
- **What $D_{\text{oracle}}(i, j)$ actually is:** It is the **ground-truth intervention damage (Oracle KL divergence)** from physically merging expert $i$ with training expert $j$ on 512-token calibration sequences.
- **The "Capability Geometry" is derived from physical Oracle interventions:**
  To place test expert $i$ into the 4D space, the experiment **must evaluate the physical merge damage between expert $i$ and all 43 training experts**.
- **Cost Paradox:** In each fold, evaluating all 21 test experts against 43 training experts requires:
  $$21 \times 43 = \mathbf{903\text{ exact Oracle merge evaluations}}$$
  This is over **4× larger** than the entire test set being predicted ($\binom{21}{2} = \mathbf{210\text{ test pairs}}$).
- **Misalignment with CARE-COM Online Framework:**
  In the sequential compression paper (CARE-COM online algorithm), capability geometry is claimed to be an *inexpensive pre-merge observational probe* ($C_t \in \mathbb{R}^{10}$) computed from forward activations without performing merges. But in Experiment 4 (Table 1), "Capability Geometry" is an **inductive low-rank approximation of the physical Oracle damage matrix itself**.
- **Audit Verdict:** The paper must explicitly clarify that Experiment 4 validates the **low-rank compressibility of the physical intervention landscape** ($q=4$ MDS), rather than proving that lightweight activation vectors alone predict Oracle KL zero-shot.

---

## 4. Dyadic Dependence and Statistical Power

### A. The Shared-Expert Correlation Structure
- In each test fold with 21 test experts, there are 210 pair observations.
- Each expert participates in exactly 20 pairs:
  $$(e_1, e_2), (e_1, e_3), \dots, (e_1, e_{21})$$
- Pairwise observations sharing an expert are statistically dependent (dyadic clustering).
- If an expert has an unusually high base sensitivity (e.g., Expert 22 in `results/exp3b/oracle_distance_matrix_middle.csv`, whose merge KL is consistently $> 0.03$ across all partners), all 20 pairs involving that expert will simultaneously have high Oracle KL and high geometric distance.
- Treating all 3,255 pairs in `experiment4_pair_level.csv` as independent i.i.d. draws produces a deceptively small p-value ($p < 10^{-300}$), artificially inflating perceived statistical power.

### B. Independent Statistical Units ($N = 5$)
- The original authors correctly recognized this in [`results/exp4/final_report.md:L164-169`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/final_report.md#L164-L169):
  > *"The 5 partitions are the independent statistical units. The 3 folds within each partition are correlated. Bootstrap CI reflects sampling variability over N=5 units only. These results must NOT be interpreted as high-power inference."*
- **Audit Assessment:** At $N=5$ independent partitions, the Wilcoxon signed-rank test yields $p = 0.0312$ (the minimum possible p-value for $N=5$ two-tailed). While statistically significant at $\alpha = 0.05$, this sample size provides limited statistical power. The paper must preserve the cautionary note on $N=5$ rather than presenting the result as a large-scale asymptotic certainty.

---

## 5. Precision@K and Parameter Budget Disambiguation

The paper conflates the letter **$K$** across three unrelated algorithmic concepts:
1. **$K$ in Table 1 (Precision@K):** Evaluates ranking precision for top candidate retrieval ($K \in \{10, 25, 50\}$). A predicted pair is deemed a true positive if it falls in the top-$K$ lowest Oracle KL pairs within the 210-pair fold test set.
2. **$K$ in Online Sequential Compression:** Represents the computational candidate evaluation budget ($K=5$ in `care_com.py`, $K=8$ in `v2.0` specification).
3. **$q$ in Multidimensional Scaling:** The geometric embedding dimensionality ($q=4$).

### Required Clarification in Manuscript:
The paper should replace the offline metric notation with **$\text{Prec@Top-}k$** or **$\text{P@}M$** ($M \in \{10, 25, 50\}$) to avoid misleading reviewers into believing that Table 1 evaluates the online $K=5$ candidate search budget.
