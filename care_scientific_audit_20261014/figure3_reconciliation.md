# Task 1 Audit Report: Figure 3 and Near-Equal-Distance Reconciliation

**Audit Date:** October 14, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifact Analyzed:** Figure 3, `audit/CLAIM_AUDIT.md`, `audit/controlled_first_step_validation.md`, `audit/experiment4_pair_level.csv`, and associated assets.

---

## 1. Executive Statement of the Discrepancy

In the submitted manuscript and supporting audit documents (`audit/CLAIM_AUDIT.md:L23-26`), two seemingly contradictory empirical statements are presented together:

1. **The 186-Pair Statistical Distribution:**  
   *"A granular analysis of 186 candidate pairs with nearly identical capability distances ($|\Delta d| < 10^{-4}$ around the median) reveals that their actual Oracle KL divergence varies materially, ranging from a minimum of 0.001522 to a maximum of 0.007096 (a ~4.6× variance in functional damage for the 'same' capability distance)."*
2. **The Illustrative Example Pairs:**  
   *"Candidate [8, 13] (distance 0.0156) and Candidate [13, 15] (distance 0.0157) are geometrically indistinguishable, yet result in materially different KL damages (0.074 vs 0.046)."*

### The Apparent Contradiction
- If the 186 candidate pairs span an Oracle KL range strictly bounded between **0.001522** and **0.007096**, the example pairs cannot belong to this set because their reported KL values (**0.074** and **0.046**) are approximately **10× to 48× larger** than the upper bound ($0.007096$).
- Furthermore, the capability distances of the example pairs ($0.0156$ and $0.0157$) are roughly **7× larger** than the median around which the 186 pairs were sampled ($0.002214$).

---

## 2. Forensic Traceability & Source Provenance

### A. Provenance of the 186-Pair Dataset
- **Primary Source File:** [`audit/experiment4_pair_level.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/experiment4_pair_level.csv) (3,255 rows).
- **Generation Script:** [`reconstruct_exp4.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/reconstruct_exp4.py).
- **Origin Experiment:** Experiment 4 (5 partitions × 3 folds expert-disjoint cross-validation on OLMoE-1B-7B middle-layer embeddings, evaluated against offline Oracle KL targets).
- **Selection Formula:**
  $$\text{median}(d_C) = 0.0022137233$$
  $$\text{Subset} = \left\{ (i, j) \in \text{Exp4 Pairs} \;\middle|\; |d_C(i, j) - 0.0022137233| < 10^{-4} \right\}$$
- **Exact Verification:**
  - Applying this exact predicate yields **exactly 186 rows**.
  - Minimum Oracle KL in subset: **0.0015228844** (Pair `partition_03_fold_01_1_19`).
  - Maximum Oracle KL in subset: **0.0070960530** (Pair `partition_00_fold_01_18_37`).
  - Ratio: $\frac{0.0070960530}{0.0015228844} = 4.6596\dots \approx \mathbf{4.66\times}$.
- **Verification Status:** **100% Mathematically Verified**. The 186 pairs and the $[0.001522, 0.007096]$ interval are exact.

### B. Provenance of the Example Pairs [8, 13] and [13, 15]
- **Primary Source File:** [`audit/controlled_first_step_validation.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/controlled_first_step_validation.md#L7-L8).
- **Origin Context:** Initial exploratory execution of sequential expert consolidation on `allenai/OLMoE-1B-7B-0924` (Step 1: 64 $\to$ 63 experts).
- **Verification against `experiment4_pair_level.csv`:**
  - Pair `(8, 13)` in Exp 4 appears in Layer 1 ($d_C = 0.000730, \text{KL} = 0.000942$) and Layer 4 ($d_C = 0.000698, \text{KL} = 0.000942$).
  - Pair `(13, 15)` in Exp 4 appears in Layer 1 ($d_C = 0.003252, \text{KL} = 0.002964$).
  - Neither pair in Experiment 4 matches $d_C \approx 0.0156$ nor $\text{KL} \in \{0.046, 0.074\}$.

---

## 3. Root Cause Analysis of the Discrepancy

The mismatch is **not** a numerical calculation error or a random hallucination. It is an editorial **cross-experimental conflation**:

1. **Two Distinct Experimental Regimes:**
   - **Regime 1 (Offline Metric MDS on Middle Layer):** Experiment 4 extracted representations normalized per layer in an offline 4D MDS space (`seq_len=512`), resulting in typical distances $\sim 0.001–0.003$ and localized Oracle KL damages $\sim 0.001–0.008$.
   - **Regime 2 (Online End-to-End Sequential Compression):** The online benchmark (`care_com.py`) evaluated full-model autoregressive forward passes on 128 calibration sequences over the unpartitioned network, producing distance metrics $\sim 0.015–0.025$ and end-to-end marginal KL divergences $\sim 0.04–0.40$.
2. **The Conflation in Text:**
   When drafting Section 3 and the accompanying claim audits, the authors merged two separate supporting arguments into a single bullet point:
   - *Argument A (Statistical Population):* The 186-pair analysis from Experiment 4.
   - *Argument B (Concrete Step-1 Instance):* An illustrative candidate pair comparison from an online sequential trial.
3. **Reviewer Impact:**
   A reader or reviewer examining Figure 3 and reading the text reasonably infers that Candidates `[8, 13]` and `[13, 15]` are two specific points drawn from the 186-pair distribution plotted in Figure 3. Because their numbers are an order of magnitude larger than the plot axes, this creates an immediate perception of fabricated or misattributed data.

---

## 4. Exact Corrected Statistics and Internal Example Pairs

To eliminate this discrepancy without modifying submitted files, we reconstruct the exact statistics and identify genuine illustrative pairs **drawn strictly from within the verified 186-pair subset** of `audit/experiment4_pair_level.csv`:

### Corrected 186-Pair Statistical Distribution
| Metric | Value |
|---|---|
| Sample Size ($N$) | 186 candidate pairs |
| Capability Distance Band | $[0.002114, 0.002314]$ ($|\Delta d| < 10^{-4}$) |
| Median Distance | 0.002214 |
| Min Oracle KL | 0.001523 |
| Max Oracle KL | 0.007096 |
| Mean Oracle KL | 0.002403 |
| Std Dev Oracle KL | 0.000968 |
| Interquartile Range (IQR) | $[0.001900, 0.002413]$ |
| Damage Ratio ($\max / \min$) | **4.66×** |

### Valid Internal Example Pairs (From Within the 186-Pair Subset)
The following two pairs exist within the 186-pair subset, share nearly identical capability distances ($|\Delta d| < 3 \times 10^{-6}$), yet exhibit a **4.23× difference in functional damage**:

1. **Pair A (`partition_01_fold_02_18_37`):**
   - Capability Distance: **0.00215659**
   - Oracle KL Damage: **0.00709638**
2. **Pair B (`partition_03_fold_00_8_57`):**
   - Capability Distance: **0.00215947**
   - Oracle KL Damage: **0.00167638**

$$\Delta d_C = |0.00215659 - 0.00215947| = 0.00000288 \quad (< 3 \times 10^{-6})$$
$$\text{Damage Ratio} = \frac{0.00709638}{0.00167638} = \mathbf{4.23\times}$$

---

## 5. Audit Recommendation for Paper Errata / Rebuttal

- **Action:** Issue an erratum clarifying that Candidates `[8, 13]` and `[13, 15]` were drawn from an online step-1 calibration run rather than the offline 186-pair Experiment 4 dataset.
- **Substitution:** Update the illustrative callout in Section 3 to reference **Pair `[18, 37]` vs. Pair `[8, 57]`** (or report both contexts with explicitly distinct labels).
- **Substantive Claim Validity:** **CLAIM B SURVIVES.** The core scientific claim—that near-equal capability distance does not guarantee identical functional consequence, varying by over 4×—is **fully substantiated** by the verified 186-pair distribution.
