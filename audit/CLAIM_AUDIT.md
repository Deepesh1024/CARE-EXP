# LOAD-BEARING CLAIM AUDIT

Based on the pair-level reconstruction and empirical re-evaluation of Experiment 4 and the v2.1 benchmark, here are the classifications of the scientific claims.

---

### CLAIM A: "Functional geometry contains predictive information about merge damage."
**Classification: SUPPORTED**

**Evidence:**
- The empirical mean per-fold Spearman correlation between capability distance and actual Oracle KL damage is `0.7504`.
- This substantially outperforms the local baseline correlation of `0.4797`.
- The Combined model achieves an even higher correlation of `0.8146`.
- At K=50, candidate pairs chosen by geometry yield a cumulative Oracle KL of `0.064`, significantly outperforming the local baseline's `0.077`.
- Furthermore, 68% of the top-50 pairs selected by geometry fall below the strict acceptable-damage threshold, compared to only 28% for the local baseline.

---

### CLAIM B: "Capability similarity does not uniquely determine intervention consequence."
**Classification: SUPPORTED**

**Evidence:**
- Despite the strong global correlation (Pearson R = 0.98), significant local variance exists. 
- A granular analysis of 186 candidate pairs with nearly identical capability distances (`|d_i - d_j| < 1e-4` around the median) reveals that their actual Oracle KL divergence varies materially, ranging from a minimum of `0.001522` to a maximum of `0.007096` (a ~4.6x variance in functional damage for the "same" capability distance).
- The controlled first-step validation confirms this natively: Candidate `[8, 13]` (distance 0.0156) and Candidate `[13, 15]` (distance 0.0157) are geometrically indistinguishable, yet result in materially different KL damages (0.074 vs 0.046).
- Therefore, while capability similarity effectively identifies a plausible neighborhood of candidates, it cannot uniquely guarantee the lowest functional consequence without exact evaluation.

---

### CLAIM C: "Recomputing capability state after physical interventions improves compression."
**Classification: SUPPORTED**

**Evidence:**
- The strict isolation study between CARE-Static and CARE-Adaptive (starting from the identical `M_64` checkpoint and using the exact same physical merge operator) demonstrates consistent improvements when capability state is adaptively recomputed.
- Adaptive recomputation yields relative reductions in Cumulative KL Divergence of `17.8%` (at 60 experts), `24.6%` (at 56 experts), and `15.9%` (at 48 experts).
- It also systematically improves the end-to-end Perplexity (PPL) at all targeted compression ratios.

---

### CLAIM D: "Compression should generally be formulated as a sequential functional intervention problem."
**Classification: SUPPORTED**

**Evidence:**
- By synthesizing Claim B and Claim C, we see that static, pre-computed rankings are fundamentally flawed because they cannot account for the local variance in intervention consequence (Claim B) nor the shifting topology of the model post-merge (Claim C).
- Because a single static capability matrix decays in accuracy as the model is structurally altered, and because exact functional evaluation is required to disambiguate geometrically similar candidates, compression intrinsically benefits from being formulated as a sequential, adaptive functional intervention problem rather than a one-shot clustering problem.
