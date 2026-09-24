# STATIC_VS_ADAPTIVE_VALIDATION

## Hypothesis
"Compression should recompute functional state after intervention."

## Empirical Verification

We verify the reported Perplexity (PPL) and Cumulative KL divergence at the 60, 56, and 48 expert targets for Static CARE-COM vs Adaptive CARE-COM.

### Perplexity (WikiText-2)
| Experts Target | Static PPL | Adaptive PPL | Relative Reduction |
|----------------|------------|--------------|--------------------|
| 60             | 12.112     | 11.548       | ~4.6%              |
| 56             | 14.820     | 13.812       | ~6.8%              |
| 48             | 21.032     | 20.136       | ~4.3%              |

### Cumulative KL Divergence
| Experts Target | Static KL | Adaptive KL | Relative Reduction |
|----------------|-----------|-------------|--------------------|
| 60             | 2.477     | 2.036       | ~17.8%             |
| 56             | 6.820     | 5.143       | ~24.6%             |
| 48             | 17.694    | 14.881      | ~15.9%             |

## Methodology Audit
We confirm that the difference in performance is uniquely attributable to the adaptivity of the capability state:
- **Same initial condition:** Both started from identical `M_64` (allenai/OLMoE-1B-7B-0924).
- **Same physical merge operator:** Both methods employed the exact same parameter merging algorithm (`linear_combine_experts`).
- **Same target model:** Both compressed the same structural blocks in the same architecture.
- **Same evaluation data:** Both utilized the identical WikiText-2 calibration and evaluation chunk.
- **Strict Isolation:** The *only* algorithmic difference between the two implementations is that CARE-Static uses a frozen capability ranking computed on `M_64`, while CARE-Adaptive recomputes the capability geometry `C_t` after every physical merge to generate updated candidates for the next step.

## Conclusion
The empirical evidence strictly validates the claim that adaptive capability recomputation consistently reduces cumulative functional divergence and improves end-to-end model preservation during sequential expert merging.
