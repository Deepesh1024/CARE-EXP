# Phase 4: Leakage-Free Geometry Validation & Experiment 4 Audit

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 18, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifact Location:** [`audit_repair_pass/pre_review_sprint/GEOMETRY_LEAKAGE_AUDIT.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/pre_review_sprint/GEOMETRY_LEAKAGE_AUDIT.md)  

---

## 1. Executive Summary: The Transductive Oracle Leakage Finding

The central empirical claim in Section 3 of the submitted manuscript asserts:
> *"Capability geometry achieves a mean held-out Spearman correlation of 0.7504 with observed merge damage on OLMoE-1B-7B-0924, versus 0.4797 for a local descriptor baseline."* (Abstract, Claim 1, Table 1, Figure 2)

Our forensic investigation of the data pipeline behind Experiment 4 ([`experiments/experiment4/`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4/)) confirms the mathematical arithmetic of the reported statistics ($\rho = 0.7504$, $\rho_{\text{pair}} = 0.7657$), but uncovers a **fatal methodological contamination and conceptual misalignment**:

1. **The Test Expert Coordinates Were Fit to Oracle Merge Damages:**
   In [`experiments/experiment4/mds_embedding.py:L114-150`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4/mds_embedding.py#L114-L150), held-out test experts were **not** embedded using pre-merge forward activations. Instead, for each test expert $i \in \text{test}$, its 4D coordinate $z_i \in \mathbb{R}^4$ was optimized via L-BFGS-B to minimize stress against its **ground-truth Oracle merge KL divergences** with all 43 training experts:
   $$z_i = \arg\min_z \sum_{j \in \text{train}} \left( \|z - Z_j\|_2 - D_{\text{oracle}}(i, j) \right)^2$$
2. **Oracle Evaluation Cost Inversion:**
   To embed the 21 held-out experts in a test fold, the algorithm physically executed $21 \times 43 = \mathbf{903\text{ exact Oracle merge evaluations}}$. This is over **4× larger** than the entire test set being predicted ($\binom{21}{2} = \mathbf{210\text{ test pairs}}$).
3. **Misalignment with the CARE-COM Online Framework:**
   In CARE-COM (Section 5), "capability geometry" is an **inexpensive observational probe representation** $C_i = F(e_i; \mathcal{P})$ computed from model forward passes with **zero** merge interventions. But in Experiment 4 (Table 1), "geometry" is an **inductive low-rank metric MDS factorization of the ground-truth physical merge damage matrix itself**.
4. **Audit Verdict:**
   Experiment 4 proves that the physical Oracle intervention landscape has low intrinsic dimensionality ($q=4$ MDS achieves $R^2 \approx 0.75$). **It does NOT prove that pre-merge activation probe vectors predict merge damage out-of-sample.** Claim 1 as framed in the manuscript conflates an offline low-rank matrix decomposition with online zero-shot observational prediction.

---

## 2. Systematic Forensic Data Flow Audit

### A. Identification of Partitions, Folds, and Units
- **Partition Design:** 5 independent partitions $\times$ 3 folds = 15 evaluation folds.
- **Expert Allocation:** In each partition, the 64 experts of `allenai/OLMoE-1B-7B-0924` (middle layer) are partitioned into 3 disjoint subsets of size 21, 21, and 22.
- **Training Units:** $N_{\text{train}} \approx 42\text{ or }43$ experts; training pairs require $i, j \in \text{train}$ ($\binom{43}{2} = 903$ pairs).
- **Test Units:** $N_{\text{test}} \approx 21\text{ or }22$ experts; test pairs require $i, j \in \text{test}$ ($\binom{21}{2} = 210$ pairs).
- **Cross-pairs:** Pairs with $i \in \text{train}, j \in \text{test}$ are excluded from Model A/C training and final evaluation.

---

### B. Trace of Oracle Value Usage

| Stage of Experiment 4 | Data Source Used | Exact Target Variable | Was Held-Out Information Used? |
|---|---|---|:---:|
| **1. Training MDS Embedding** | `results/exp3b/oracle_distance_matrix_middle.csv` | Train$\times$Train Oracle KL ($\binom{43}{2} = 903$ pairs) | **NO** (Strictly within training partition) |
| **2. Out-of-Sample Test Embedding** | `results/exp3b/oracle_distance_matrix_middle.csv` | Test$\times$Train Oracle KL ($21 \times 43 = 903$ pairs) | **YES** (Test expert merge outcomes fit $z_i$) |
| **3. Test Pair Prediction** | Distance $\|z_i - z_j\|_2$ | Predicted KL damage for test pair $(i, j)$ | **Derived from Step 2** |
| **4. Test Evaluation** | `results/exp3b/oracle_distance_matrix_middle.csv` | Test$\times$Test Oracle KL ($\binom{21}{2} = 210$ pairs) | Target evaluation only (Unseen in Step 2) |

---

### C. Analysis of the Five Influence Vectors

1. **Embedding Optimization:**  
   **CONTAMINATED.** The test expert coordinates $z_i$ and $z_j$ were directly fit to match their Oracle merge KLs with training experts. While the specific test-test pair damage $D_{\text{oracle}}(i, j)$ was omitted during optimization, both $z_i$ and $z_j$ were already informed by 43 physical merge experiments involving those exact experts.
2. **Feature Selection:**  
   Local descriptors for Model A and Model C (11 features: weight norms, cosine similarities, routing entropy) were extracted purely from pre-merge checkpoints.
3. **Hyperparameter Selection:**  
   The embedding dimension $q=4$ was selected based on reconstruction stress of the full 64×64 Oracle damage matrix from Experiment 3B.
4. **Stopping Criteria:**  
   L-BFGS-B optimization terminated when the gradient of stress with respect to test-train Oracle distances reached machine epsilon (`gtol=1e-5`).
5. **Candidate Ranking:**  
   Ranking candidates by $\|z_i - z_j\|_2$ is not an observational ranking; it is a matrix completion ranking derived from 903 prior Oracle interventions.

---

## 3. Dyadic Dependence and Statistical Power Analysis

The manuscript reports pair-level correlation across $N = 3,255$ pooled pairs:
$$\text{Spearman } \rho = 0.7657 \quad (N = 3,255)$$

### The Dyadic Clustering Confound:
1. In a fold with 21 test experts, there are 210 pair observations. However, each expert participates in exactly 20 pairs: $(e_1, e_2), (e_1, e_3), \dots, (e_1, e_{21})$.
2. If expert $e_1$ has a high baseline sensitivity (e.g., an indispensable central expert), all 20 pairs containing $e_1$ will simultaneously exhibit high Oracle damage and high geometric distance.
3. Observations sharing an expert are statistically dependent. Treating all 3,255 observations as independent draws falsely collapses standard errors ($p < 10^{-300}$).
4. The true independent statistical units are the **$N = 5$ cross-validation partitions**. Across $N = 5$ independent partitions, the mean held-out Spearman $\rho$ is:
   - Model A (Local Baseline): $\mathbf{0.4797 \pm 0.038}$
   - Model B (Geometry MDS): $\mathbf{0.7504 \pm 0.026}$
   - Model C (Combined): $\mathbf{0.8146 \pm 0.021}$
5. While the difference between Model B and Model A is statistically significant under a Wilcoxon signed-rank test ($W = 15, p = 0.03125$ for $N=5$), the sample size ($N=5$) represents limited statistical power. Multi-partition replication is required before asserting universal generalization.

---

## 4. Controlled Leakage-Free Geometry Validation Design

To resolve the transductive contamination, we specify a clean, strictly out-of-sample evaluation protocol:

### Specification:
1. **Predictive Signal (Strictly Observational):**
   - Use the **genuine probe-based capability representation** $C_i = F(e_i; \mathcal{P})$ extracted from `EXP6C_TOKEN_CAPABILITY_VECTORS.parquet` (pre-merge forward passes on 128 probe tokens).
   - Compute pairwise capability distance: $d_C(i, j) = \|C_i - C_j\|_2$.
   - **Zero** physical merges or Oracle KL values are permitted during capability extraction or coordinate assignment.
2. **Evaluated Ground Truth:**
   - Physical merge intervention damage $D_{\text{oracle}}(i, j)$ measured on calibration sequences.
3. **Controls:**
   - Control 1: Parameter Weight Distance $\|W_i - W_j\|_F$.
   - Control 2: Parameter Cosine Distance $1 - \cos(W_i, W_j)$.
   - Control 3: Output Activation Similarity (token forward cosine).
   - Control 4: Routing Policy Similarity (router weight cosine).
4. **Statistical Accounting:**
   - Evaluate Spearman rank correlation $\rho(d, D_{\text{oracle}})$ and Precision@Top-$K$.
   - Compute cluster-robust standard errors grouped by expert identity (dyadic jackknife / cluster bootstrap).
   - Report 95% confidence intervals based on effective degrees of freedom.

---

## 5. Audit Deliverable & Required Manuscript Corrections

1. **Clarify Nature of Experiment 4:**  
   The manuscript must transparently state that Experiment 4 (Table 1, Figure 2) evaluates an **MDS low-rank embedding of the physical Oracle intervention matrix**, demonstrating that the intervention damage space has low intrinsic dimension ($q=4$).
2. **Remove Misleading "Zero-Shot Observational" Framing:**  
   Do not describe Experiment 4 as proving that lightweight probe vectors predict merge damage without prior merge evaluations.
3. **Disambiguate Candidate Budget ($K=5$) from Precision ($P@M$):**  
   Change Table 1 column labels from `Precision@10`, `Precision@25`, `Precision@50` to `P@Top-10`, `P@Top-25`, `P@Top-50` to avoid conflating the evaluation metric with the online candidate budget $K=5$.
