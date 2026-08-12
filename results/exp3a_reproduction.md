# EXPERIMENT 3A: REPRODUCTION AUDIT

## Overview
We independently recalculated the primary H2 statistic: the difference in Oracle KL between within-community merges and between-community merges.
This relies on the pre-computed community assignments from Experiment 3A (`k=8`).

## Reproduction Results

### Layer: First
- **N Within**: 51
- **N Between**: 1965
- **Mean Within KL ($D_{within}$)**: 0.003314
- **Mean Between KL ($D_{between}$)**: 0.005107
- **T = $D_{between} - D_{within}$**: 0.001793
- **Ratio ($D_{between} / D_{within}$)**: 1.54
- **Mann-Whitney U p-value**: 3.7775e-13

### Layer: Middle
- **N Within**: 139
- **N Between**: 1877
- **Mean Within KL ($D_{within}$)**: 0.002050
- **Mean Between KL ($D_{between}$)**: 0.003717
- **T = $D_{between} - D_{within}$**: 0.001667
- **Ratio ($D_{between} / D_{within}$)**: 1.81
- **Mann-Whitney U p-value**: 1.0535e-10

### Layer: Last
- **N Within**: 117
- **N Between**: 1899
- **Mean Within KL ($D_{within}$)**: 0.001863
- **Mean Between KL ($D_{between}$)**: 0.005457
- **T = $D_{between} - D_{within}$**: 0.003594
- **Ratio ($D_{between} / D_{within}$)**: 2.93
- **Mann-Whitney U p-value**: 4.0309e-54

### Layer: Aggregated
- **N Within**: 115
- **N Between**: 1901
- **Mean Within KL ($D_{within}$)**: 0.003244
- **Mean Between KL ($D_{between}$)**: 0.004722
- **T = $D_{between} - D_{within}$**: 0.001478
- **Ratio ($D_{between} / D_{within}$)**: 1.46
- **Mann-Whitney U p-value**: 6.8729e-24

## Conclusion
**REPRODUCED**: We exactly match the results in the original `experiment3a_report.md`.
The difference between within-community and between-community Oracle KL is real and statistically significant.
However, as noted in the Provenance Audit, this was calculated using communities derived from a Surrogate predictor trained on the same data, creating severe circularity.