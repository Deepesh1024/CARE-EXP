# Pre-Registration Plan: Full 64-Expert Matched Rematch

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 19, 2026  
**Status:** PREREGISTERED FOR POST-AUDIT EXECUTION  
**Artifact:** [`audit_repair_pass/random_k5_forensic_audit/NEXT_EXPERIMENT_PREREGISTRATION.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/random_k5_forensic_audit/NEXT_EXPERIMENT_PREREGISTRATION.md)

---

## 1. Scientific Objective & Rationale

Our forensic audit discovered two foundational facts:
1. **The KL/PPL Discrepancy is Resolved:** Greedy candidate selection on 128 final-token predictions (`outputs.logits[:, -1, :]`) allows Random-$K=5$ to achieve near-zero marginal KL by picking inactive expert pairs, but causes catastrophic cross-entropy explosion (+9.68 PPL) on the full 65,536-token test set. Capability geometry acts as a global structural regularizer that prevents this out-of-distribution failure.
2. **The Historical Asymmetry:** The historical CARE-Adaptive benchmark run inadvertently restricted candidate nominations to expert indices $[0, 15]$ due to an indexing artifact in `capability.py`, while the audited Random-$K=5$ runs searched across all 64 experts $[0, 63]$. Furthermore, CARE-Adaptive was only run on 1 seed, preventing a true paired statistical test.

To provide an incontrovertible scientific evidence package for ICLR 2027 reviewers, we pre-register the **Full 64-Expert Matched Rematch**.

---

## 2. Experimental Specification

### 2.1. Exact Model and Precision
- **Model Checkpoint:** `allenai/OLMoE-1B-7B-0924` (HuggingFace Hub)
- **Base Expert Count:** 64 physical experts per layer across 16 layers (Top-8 routed).
- **Precision:** Native `bfloat16` with deterministic PyTorch operations where available.
- **Device:** Remote NVIDIA L40S GPU (48 GB GDDR6 VRAM, Host: `ubuntu@45.194.3.117`).

### 2.2. Compression Trajectory & Search Budget
- **Compression Schedule:** 64 $\to$ 60 $\to$ 56 $\to$ 48 experts (16 sequential merge steps).
- **Candidate Evaluation Budget:** Exactly $K = 5$ candidate merges evaluated per step ($80$ physical interventions per run).
- **Objective Function:** Marginal KL divergence over 128 calibration chunks (`evaluate_marginal_kl`).
- **Nomination Space:** **Full active expert set $[0, N_t - 1]$** (all 64 experts profiled and eligible for nomination at Step 1).
- **Router Combination Rule:** $(W_{r, i} + W_{r, j}) / 2.0$.

### 2.3. Paired Seed Configuration ($N = 5$)
To compute a true paired $t$-test and true paired non-parametric tests, CARE-Adaptive will be executed across the exact 5 random seeds used for the Random-$K=5$ runs:
- **Seed Set:** $S = \{101, 202, 303, 404, 505\}$.
- **Calibration Draw:** Each seed $s \in S$ uses `df_calib = df_full.sample(n=128, random_state=s)`.
- **Paired Matching:** CARE-Adaptive(Seed $s$) will be paired directly against Random-$K=5$(Seed $s$), ensuring identical calibration token subsets for both algorithms within each pair.

---

## 3. Computational Budget and Hardware

| Parameter | Estimate |
|:---|:---|
| **Per-Step Wall Time** | ~40 seconds (profiling 64 experts + 5 snapshot/restore evaluations) |
| **Per-Run Wall Time (16 steps + PPL checkpoints)** | ~680 seconds (~11.3 minutes) |
| **Total Wall Time for 5 Seeds** | $\approx 56.5\text{ minutes}$ |
| **Peak VRAM Allocated** | $\le 26.5\text{ GB}$ (well within the 48 GB L40S envelope) |
| **Compute Cost** | $\approx \$1.20$ USD on standard cloud GPU instances |

---

## 4. Stopping Rules and Algorithmic Guardrails

1. **Numerical Validity Invariant:** If `evaluate_marginal_kl` returns `NaN` or `Inf` on any candidate, or if test PPL returns `NaN`, execution immediately aborts with error logging.
2. **State Leakage Invariant:** Layer 0 Expert 0 Gate SHA-256 tensor hash must be verified after every snapshot restore. If restoration fails to match the pre-evaluation hash, the run halts immediately.
3. **Atomic Persistence:** Each step trace is written atomically to JSON to eliminate log truncation or buffer corruption.

---

## 5. Pre-Registered Decision Boundaries

Before executing any code, we establish the explicit scientific interpretation boundaries for the final WikiText-2 test perplexity at 48 experts ($\text{PPL}_{48\text{E}}$):

```
                   Decision Boundary Spectrum (PPL @ 48 Experts)
  
  [--- DECISIVE SUCCESS ---]      [-- QUALIFIED MERIT --]       [--- CLAIM FALSIFIED ---]
 10.28 (Base)            22.00   22.00                 26.00   29.81 (Rand-5 Mean)     35.0+
   |-----------------------|-------|---------------------|-------|-----------------------|
   ^ Full 64E CARE beats   ^       ^ Statistically       ^       ^ CARE is no better than
   every random seed with          better than Rand-5,           random search under equal
   p < 0.01 (paired t-test)        but search gap shrinks        budget. Claim rejected.
```

### Boundary A: Decisive Confirmation of Algorithmic Superiority
- **Criteria:**
  1. Mean CARE-Adaptive $\text{PPL}_{48\text{E}} \le 22.00$.
  2. CARE-Adaptive beats Random-$K=5$ on **all 5 paired seeds** ($\Delta_i > 0$ for all $i$).
  3. Paired $t$-test is statistically significant at $\alpha = 0.01$ ($t \ge 4.604, p < 0.01, df = 4$).
- **Scientific Impact:** Completely validates that capability geometry is a robust, unassailable global regularizer across the full 64-expert space. Paper revision can strongly assert superiority over search-matched baselines.

### Boundary B: Qualified Architectural Merit
- **Criteria:**
  1. Mean CARE-Adaptive $\text{PPL}_{48\text{E}}$ falls in $(22.00, 26.00]$.
  2. CARE-Adaptive beats Random-$K=5$ mean ($29.81$) with $p < 0.05$ on a paired $t$-test.
- **Scientific Impact:** Confirms capability geometry outperforms random search under matched budgets, but acknowledges that global multi-layer routing interactions across 64 unconstrained experts degrade slightly more than localized clustering.

### Boundary C: Falsification of Nomination Superiority
- **Criteria:**
  1. Mean CARE-Adaptive $\text{PPL}_{48\text{E}} \ge 28.00$, OR
  2. Paired $t$-test fails to reject $H_0$ at $\alpha = 0.05$ ($p \ge 0.05$).
- **Scientific Impact:** Falsifies the claim that geometric nomination provides substantial advantage over random candidate evaluation under matched search budgets. Manuscript must be reframed to highlight search-budget dynamics as the primary driver of compression quality.
