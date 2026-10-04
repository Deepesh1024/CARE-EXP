# Experiment 8: GWS Validity and Low-Damage Regime Resolution
**Status:** CLOSED
**Evaluated Substitutions:** 1,750 (Sub-sampled FP32 subset of the 4,032 ordered expert pairs in Layer 8 of OLMoE-1B-7B)

## 1. Primary Pre-Registered Result
**Question:** Does the local, first-order Gradient-Weighted Substitution (signed GWS) score reliably predict the finite functional substitution damage caused by swapping one expert for another on held-out text?

**Criteria for Validity:** $\rho_{GWS} \ge 0.50$

**Result:**
| Predictor | Spearman ρ | Clustered Bootstrap 95% CI |
| :--- | :--- | :--- |
| **GWS signed** | 0.148 | [-0.198, 0.436] |

**Conclusion:** GWS fails the pre-registered validity gate and will not be incorporated into CARE. The experiment does not establish a reliable positive rank correlation for GWS across source-expert clusters, and the observed correlation is far below the pre-specified usefulness threshold. 

---

## 2. Low-Damage Regime Resolution
By casting the logits to FP32 immediately before cross-entropy evaluation, the mathematical quantization floor was successfully lifted. 
- **1,750** substitutions were evaluated in the current FP32-resolved analysis.
- **150 / 1750 (8.6%)** substitutions actually improved held-out CE (negative $\Delta$CE).
- **Minimum $\Delta$CE:** -0.0002237
- **0 / 1750** pairs were obscured by the $10^{-7}$ noise floor.

This confirms we are measuring a true, high-resolution finite intervention landscape rather than getting a monotonic "replacement = damage" artifact.

---

## 3. Secondary Predictor Comparison
| Predictor | Spearman ρ | Clustered Bootstrap 95% CI |
| :--- | :--- | :--- |
| **GWS squared** | 0.310 | [-0.025, 0.581] |
| **Usage** | 0.766 | [0.468, 0.922] |
| **RW-L2** | 0.838 | [0.653, 0.940] |

*Note: The Clustered 95% CIs for Usage and RW-L2 overlap substantially (`[0.468, 0.922]` vs `[0.653, 0.940]`), meaning we cannot formally claim from these correlations alone that RW-L2 dominates Usage.*

---

## 4. OOS Incremental Analysis (GroupKFold over source expert $j$)
To determine if these signals capture overlapping structure, we computed Out-of-Sample (OOS) $R^2$ using 5-fold cross-validation clustered by source expert:

| Model | Fold OOS $R^2$ | Mean ± Std $R^2$ |
| :--- | :--- | :--- |
| **A: Usage** | `[ 0.793  0.303 -0.313 -0.442  0.682]` | **0.2044 ± 0.5037** |
| **B: Usage + RW-L2** | `[ 0.921  0.373  0.021 -0.491  0.597]` | **0.2845 ± 0.4860** |
| **C: Usage + RW-L2 + signed GWS** | `[ 0.786  0.450  0.432  0.041  0.637]` | **0.4694 ± 0.2506** |
| **D: Usage + RW-L2 + GWS²** | `[ 0.920  0.359 -0.807 -0.497  0.593]` | **0.1136 ± 0.6571** |

**Partial Correlations (Clustered Bootstrap 95% CI):**
- **RW-L2 \| Usage:** 0.151 `[-0.202, 0.642]`
- **Usage \| RW-L2:** 0.344 `[0.093, 0.653]` *(Statistically Significant)*

**Conclusions on Secondary Predictors:**
1. **Usage captures the underlying structure:** The partial correlation for Usage controlling for RW-L2 does not cross zero (`[0.093, 0.653]`), while RW-L2 controlling for Usage does (`[-0.202, 0.642]`). This suggests that Usage alone captures much of the predictive structure that appears in RW-L2.
2. **GWS Incremental Value:** While Model C (signed GWS) showed an unexpectedly stabilized and higher Mean OOS $R^2$ across the folds compared to Model B, Model D (GWS²) heavily deteriorated OOS performance. Given that the clustered partial correlation of `GWS | Usage` (`[-0.111, 0.592]`) comfortably crosses zero, there is insufficient evidence to conclude that local gradient projection provides reliable structural information out-of-sample.

---

## 5. Strategic Takeaway
Finite substitution damage $\neq$ local gradient projection. 
The observed damage is strongly associated with simpler quantities such as expert usage and functional displacement. The most scientifically fruitful path forward for CARE is to explicitly model the functional state/distance of the experts rather than assuming local optimization sensitivity is the right proxy for macroscopic intervention damage.
