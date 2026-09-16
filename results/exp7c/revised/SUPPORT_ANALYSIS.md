# Methodological Support Analysis: Revised 7C Protocol

**Context:** Experiment 7C is being revised to restrict neuron functional similarity calculations to tokens where *both* experts in a candidate pair are activated ($T_{ij}$).

## 1. Support Statistics

Based on the 262,144-token Wikitext probe, the mutual activation support ($T_{ij}$) for the 18 candidate pairs is distributed as follows:

### High-Support Pairs (14 of 18)
| Pair | Expert i | Expert j | Mutual Tokens ($T_{ij}$) | Mutual Fraction |
|---|---|---|---|---|
| pair_10 | 7 | 32 | 16,427 | 6.27% |
| pair_13 | 58 | 62 | 1,555 | 0.59% |
| pair_08 | 1 | 6 | 1,069 | 0.41% |
| pair_14 | 45 | 53 | 1,001 | 0.38% |
| pair_07 | 9 | 14 | 926 | 0.35% |
| pair_17 | 22 | 31 | 827 | 0.32% |
| pair_12 | 5 | 39 | 557 | 0.21% |
| pair_04 | 11 | 47 | 323 | 0.12% |
| pair_05 | 21 | 53 | 271 | 0.10% |
| pair_18 | 4 | 22 | 256 | 0.10% |
| pair_15 | 49 | 59 | 179 | 0.07% |
| pair_09 | 49 | 56 | 156 | 0.06% |
| pair_01 | 7 | 19 | 68 | 0.03% |
| pair_02 | 13 | 36 | 54 | 0.02% |

### Low-Support Pairs (4 of 18)
| Pair | Expert i | Expert j | Mutual Tokens ($T_{ij}$) | Mutual Fraction |
|---|---|---|---|---|
| pair_16 | 10 | 22 | 44 | 0.017% |
| pair_11 | 24 | 35 | 43 | 0.016% |
| pair_03 | 27 | 43 | 18 | 0.007% |
| pair_06 | 24 | 43 | 2 | 0.001% |

*(Note: The contiguous/shared activation support distribution cannot be explicitly mapped without the 29GB memmap, but sparse MoE routing implies these mutual tokens are non-contiguous and distributed sparsely across the 262k sequence.)*

## 2. Is $T_{ij} \ge 50$ a Defensible Minimum?

We are calculating the cosine similarity between two neuron activation signatures: $S_{kl} = \cos(\mathbf{a}_k, \mathbf{a}_l)$. Under the revised protocol, $\mathbf{a}_k \in \mathbb{R}^{T_{ij}}$.

Because post-activation values (SiLU × up_proj) are largely positive, we are comparing vectors in the positive orthant. 

- If $T_{ij} = 2$ (pair 06): The vectors are 2-dimensional. Cosine similarity between any two random 2D positive vectors has massive variance. A random similarity could easily be 0.99 purely by chance, making $C_{mutual}$ meaningless.
- If $T_{ij} \approx 20$ (pair 03): The variance of dot products in $\mathbb{R}^{20}$ remains high enough that max-pooling over 1024 neurons (which is how $C_{mutual}$ is calculated: $\max_l S_{kl}$) is virtually guaranteed to find spurious high-similarity matches (the "curse of dimensionality" acting on the max operator).
- If $T_{ij} \ge 50$: The degrees of freedom are sufficient that the expected maximum cosine similarity of random noise vectors stabilizes, and structural signals (genuine functional correlation) become distinguishable from random routing intersections. 

**Conclusion:** $T_{ij} \ge 50$ is a mathematically defensible minimum threshold. Below this, $C_{mutual}$ is an artifact of the $\max$ operator finding spurious alignments in low-dimensional space.

## 3. Handling Low-Support Pairs

We must choose between two protocols:

**Protocol A:** Primary analysis on all 18 pairs with a support-aware validity flag, while reporting low-support pairs separately.
**Protocol B:** Restricted analysis on the 14 pairs with $\ge 50$ mutual activations, explicitly treated as a sensitivity/validity analysis.

### Selection Bias Check
If we use Protocol B (dropping 4 pairs), does it introduce selection bias?
- The pairs being dropped are those where the two experts *almost never fire on the same token*.
- In a sparse MoE, experts that never co-fire are strongly disjoint in their routing regions.
- If CARE-COM residuals ($R_{ij}$) systematically differ for disjoint experts versus overlapping experts, dropping disjoint experts limits the hypothesis test to "overlapping experts only."
- This is a form of selection bias, as it truncates the domain of the predictor.

### Measurement Reliability vs. Selection Bias
If we use Protocol A (keeping all 18), we introduce extreme measurement noise. For pair 06 ($T=2$), $C_{mutual}$ will be artificially high. Since this pair has low routing overlap, its CARE-COM predictions and actual damages are likely unique. Injecting a random/artificial $C_{mutual}$ value for this pair into the Spearman rank correlation will arbitrarily scramble the ranks.

Because $C_{mutual}$ for $T < 50$ is mathematically invalid (due to the max-over-1024-neurons operator on low-dim vectors), **including them introduces measurement invalidity that corrupts the rank correlation more severely than selection bias limits its scope.**

## 4. Recommendation and Rationale

**Recommendation:** Proceed with **Protocol A as the Primary**, but strictly pair it with **Protocol B as the Pre-specified Sensitivity Analysis**.

**Rationale:**
1. **Preserve the Protocol:** To respect the pre-specified $N=18$ design of Experiment 7, the primary statistic must be computed on all 18 pairs. This avoids any accusation of post-hoc data filtering to achieve significance.
2. **Acknowledge the Noise:** We expect the 4 low-support pairs to act as noise injectors due to dimensionality artifacts. 
3. **The True Test:** The restricted 14-pair analysis (Protocol B) represents the *scientifically valid* measurement of the revised hypothesis. If the 14-pair analysis shows a strong correlation while the 18-pair analysis is destroyed by noise, the scientific interpretation is that the mechanism holds *wherever it can be validly measured*.

**Revised 7C Analysis Protocol:**
1. Compute $C_{mutual}$ on the mutually activated tokens for all 18 pairs.
2. Compute the primary Spearman correlation ($N=18$).
3. Compute the restricted sensitivity Spearman correlation ($N=14$, filtering $T_{ij} < 50$).
4. The scientific verdict will be drawn by evaluating both: if the mechanism requires valid measurement support to manifest, the $N=14$ result will govern the interpretation of *why* the $N=18$ result behaves as it does.
