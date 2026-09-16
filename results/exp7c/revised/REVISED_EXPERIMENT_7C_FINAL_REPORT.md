# Revised Experiment 7C: Final Report

**Date:** 2026-09-16
**Status:** COMPLETE
**Verdict:** NULL (Mechanism does not explain residuals)

## 1. Objective
Test whether the **routing-conditioned cross-expert neuron functional similarity** ($C_{mutual}$ restricted to mutually activated tokens $T_{ij}$) explains CARE-COM's residual merge error ($R_{ij}$).

## 2. Methodology Correction
The original 7C experiment computed cosine similarity over all 262,144 tokens, including unrouted and padded positions. The revised protocol:
1. Reconstructs and applies `attention_mask` to exclude padding.
2. Identifies mutually activated tokens for each pair: $T_{ij} = \{t : r_i(t) > 0 \land r_j(t) > 0\}$.
3. Computes $C_{mutual}$ exclusively on vectors in $\mathbb{R}^{T_{ij}}$.

## 3. Results

### Primary Analysis (Protocol A)
- **N = 18** (all candidate pairs)
- **Spearman $\rho$ = -0.0361**
- **p-value = 0.8869**
- **95% Bootstrap CI:** [-0.5000, +0.4630]

### Sensitivity Analysis (Protocol B)
*Restricted to pairs with $T_{ij} \ge 50$ to ensure dimensional stability of the cosine similarity.*
*(Note: After removing padded tokens, exactly 12 pairs met the $\ge 50$ threshold, down from 14 in the unmasked support audit).*
- **N = 12**
- **Spearman $\rho$ = -0.1818**
- **p-value = 0.5717**

### Comparison with Original 7C
- Original (N=18): $\rho$ = -0.0671, p = 0.791
- Revised (N=18): $\rho$ = -0.0361, p = 0.887
- **Conclusion:** The measurement correction (conditioning on routing and dropping padding) **did not materially change the result**. The correlation remains practically zero.

### Pair 7-32 Influence Diagnostic
Pair 7-32 (pair_10) had drastically higher support (16,427 mutual tokens) than all other pairs.
- Excluding pair 7-32 (N=17): $\rho$ = +0.0417, p = 0.8738
- Shift in $\rho$: $|-0.0361 - 0.0417| \approx 0.078$
- **Conclusion:** Pair 7-32 does **not** disproportionately influence the N=18 result.

## 4. Validity Concerns
- **Selection Bias (N=12):** The sensitivity analysis drops pairs that rarely co-activate. While necessary for measurement validity, it limits the test to highly overlapping experts.
- **Dimensionality Noise (N=18):** Pairs with $T_{ij} < 50$ (some as low as $T=2$) inject severe noise into the primary N=18 statistic due to the max-pooling operator over low-dimensional spaces. 
- However, since BOTH N=18 (noise-injected) and N=12 (noise-filtered, selection-biased) yielded unambiguous null results, these validity concerns do not threaten the overall negative conclusion.

## 5. Scientific Interpretation
The revised routing-conditioned mutual coverage mechanism **does not explain** CARE-COM residual error under this experiment. 

The initial null result in 7C was **not** merely an artifact of including zero-filled inactive tokens or padding. Even when strictly conditioning the neuron signatures on genuine shared expert activation, the functional geometry (as measured by nearest-neighbor cosine similarity) fails to correlate with the actual merge damage residuals. 

This suggests that while experts may have functional neuron overlap, this overlap does not predictably determine how gracefully they will merge beyond what the CARE-COM predictor already captures. CARE-COM's blind spots likely stem from a different mechanism (such as weight-space interference independent of activation similarity, or higher-order routing dynamics).
