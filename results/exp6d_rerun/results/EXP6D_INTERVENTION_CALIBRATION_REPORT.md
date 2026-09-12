# EXPERIMENT 6D: INTERVENTION CALIBRATION REPORT

## 1. ZERO-UPDATE NOISE FLOOR
- $\sigma(||\Delta C||)$: 0.000000
- $\sigma(\Delta \theta)$: 0.009485 degrees

## 2. INTERVENTION CONFIGURATION EVALUATION

### CONFIG A (Steps: 5, LR: 0.0001)
- Mean $||\Delta C||$: 0.000001 (SNR: **715630.16**)
- Mean $\Delta \theta$: 0.007505 degrees (SNR: **0.79**)
- Mean Absolute Tau Error $||\tau_{target} - \tau_{actual}||$: 0.426442
- Mean Cosine$(\tau_{target}, \tau_{actual})$: 0.996470
- Stability Exclusions ($||\Delta C|| > ||C_{before}|| / 2$): 0 / 27
- Valid Criterion Passed: **NO**

### CONFIG B (Steps: 25, LR: 0.0001)
- Mean $||\Delta C||$: 0.000003 (SNR: **3171352.77**)
- Mean $\Delta \theta$: 0.005307 degrees (SNR: **0.56**)
- Mean Absolute Tau Error $||\tau_{target} - \tau_{actual}||$: 0.426053
- Mean Cosine$(\tau_{target}, \tau_{actual})$: 0.999491
- Stability Exclusions ($||\Delta C|| > ||C_{before}|| / 2$): 0 / 27
- Valid Criterion Passed: **NO**

### CONFIG C (Steps: 50, LR: 0.0001)
- Mean $||\Delta C||$: 0.000005 (SNR: **4840148.45**)
- Mean $\Delta \theta$: 0.009703 degrees (SNR: **1.02**)
- Mean Absolute Tau Error $||\tau_{target} - \tau_{actual}||$: 0.425543
- Mean Cosine$(\tau_{target}, \tau_{actual})$: 0.999755
- Stability Exclusions ($||\Delta C|| > ||C_{before}|| / 2$): 0 / 27
- Valid Criterion Passed: **NO**

### CONFIG D (Steps: 50, LR: 0.0005)
- Mean $||\Delta C||$: 0.000051 (SNR: **50652368.37**)
- Mean $\Delta \theta$: 0.050408 degrees (SNR: **5.31**)
- Mean Absolute Tau Error $||\tau_{target} - \tau_{actual}||$: 0.425543
- Mean Cosine$(\tau_{target}, \tau_{actual})$: 0.999755
- Stability Exclusions ($||\Delta C|| > ||C_{before}|| / 2$): 0 / 27
- Valid Criterion Passed: **YES**

## 3. RECOMMENDATION
**RECOMMENDED CONFIGURATION: D**

Config D is the minimum intervention strength that satisfies both the $5\times\sigma$ independent measurement resolution criteria for functional magnitude and directional movement, while maintaining complete numerical stability within the predeclared bounds.
