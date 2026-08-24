# EXPERIMENT 6B — FINAL REPORT

### Hypothesis


**Compiled:** 2026-08-16T15:32:35.639074

### Experiment


## 5. Upstream Experiments

- Exp4 final_report.json: L
- Exp6A prediction_results.csv: L

## 6. System Resources

- CUDA: True
- GPU: NVIDIA GeForce RTX 4090
- GPU Memory: 25.3 GB


---
# Experiment 6B — Functional Alignment Report

**Generated:** 2026-08-16T15:31:59.923053
**Reference checkpoint:** checkpoint_100
**Alignment method:** Orthogonal Procrustes (no scaling)

## Layer: first

### q4 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.003679 | 0.0000 | 0.9445 | 0.9796 |
| checkpoint_40 | 0.003813 | 0.0000 | 0.9730 | 0.9826 |
| checkpoint_70 | 0.003900 | 0.0000 | 0.9775 | 0.9908 |
| checkpoint_100 (REF) | 0.000000 | 0.0020 | 0.8738 | 0.9565 |

### q6 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.003561 | 0.0000 | 0.9657 | 0.9895 |
| checkpoint_40 | 0.003618 | 0.0000 | 0.9882 | 0.9927 |
| checkpoint_70 | 0.003603 | 0.0000 | 0.9884 | 0.9958 |
| checkpoint_100 (REF) | 0.000000 | 0.0013 | 0.8895 | 0.9654 |

### q3 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.003705 | 0.0000 | 0.9136 | 0.9625 |
| checkpoint_40 | 0.003635 | 0.0001 | 0.9489 | 0.9649 |
| checkpoint_70 | 0.003566 | 0.0001 | 0.9453 | 0.9773 |
| checkpoint_100 (REF) | 0.000000 | 0.0027 | 0.8245 | 0.9373 |

## Layer: middle

### q4 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.002058 | 0.0000 | 0.9763 | 0.9996 |
| checkpoint_40 | 0.002603 | 0.0001 | 0.9672 | 0.9955 |
| checkpoint_70 | 0.003474 | 0.0001 | 0.9762 | 0.9884 |
| checkpoint_100 (REF) | 0.000000 | 0.0009 | 0.9251 | 0.9933 |

### q6 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.002071 | 0.0000 | 0.9861 | 0.9997 |
| checkpoint_40 | 0.002570 | 0.0001 | 0.9756 | 0.9959 |
| checkpoint_70 | 0.003404 | 0.0001 | 0.9818 | 0.9894 |
| checkpoint_100 (REF) | 0.000000 | 0.0007 | 0.9516 | 0.9943 |

### q3 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.002191 | 0.0000 | 0.9702 | 0.9995 |
| checkpoint_40 | 0.002762 | 0.0001 | 0.9568 | 0.9949 |
| checkpoint_70 | 0.003485 | 0.0001 | 0.9677 | 0.9873 |
| checkpoint_100 (REF) | 0.000000 | 0.0010 | 0.9131 | 0.9916 |

## Layer: last

### q4 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.003872 | 0.0001 | 0.9917 | 0.9941 |
| checkpoint_40 | 0.004261 | 0.0001 | 0.9904 | 0.9929 |
| checkpoint_70 | 0.004302 | 0.0001 | 0.9910 | 0.9916 |
| checkpoint_100 (REF) | 0.000000 | 0.0028 | 0.9353 | 0.9234 |

### q6 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.003693 | 0.0000 | 0.9942 | 0.9961 |
| checkpoint_40 | 0.003837 | 0.0001 | 0.9932 | 0.9950 |
| checkpoint_70 | 0.003933 | 0.0001 | 0.9928 | 0.9938 |
| checkpoint_100 (REF) | 0.000000 | 0.0020 | 0.9460 | 0.9376 |

### q3 (common pairs: 384, common experts: 64)

| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |
|---|---|---|---|---|
| checkpoint_10 | 0.003930 | 0.0001 | 0.9902 | 0.9932 |
| checkpoint_40 | 0.004372 | 0.0001 | 0.9852 | 0.9882 |
| checkpoint_70 | 0.004816 | 0.0002 | 0.9863 | 0.9858 |
| checkpoint_100 (REF) | 0.000000 | 0.0037 | 0.9062 | 0.8962 |



---
# Experiment 6B — Checkpoint Trajectory Analysis

### Equations


## Core Framework Definitions
To ensure clarity and distinguish observed facts from testable predictions, we establish the following framework:

- **Definition ($C_i$)**: $C_i \in \mathbb{R}^{10}$ represents the empirical capability response of expert $i$.
- **Definition ($	au_i$)**: $	au_i \in \mathbb{R}_+^{10}$ represents the local token environment presented to expert $i$.
- **Definition ($\Delta C_i$)**: $\Delta C_i = C_i(t+1) - C_i(t)$ represents the functional displacement over training interval $t 	o t+1$.
- **Definition (Decomposition)**: $\Delta C_i = \Delta C_{i, \parallel} + \Delta C_{i, \perp}$, decomposing displacement into radial (magnitude contraction/expansion) and tangential (angular/task-specific steering) components.
- **Hypothesis (Geometric Susceptibility)**: The tangential movement $\Delta C_{i, \perp}$ is directionally guided by the orthogonal component of the interaction vector $I = C_i \odot 	au_i$.

## Token/Routing Environment → Functional Evolution of MoE Experts

This report aggregates the findings from all phases of Experiment 6B, tracing how the routing environment (exposure) dictates the movement of experts through functional space across the training lifecycle.


---
# Experiment 6B  Data Audit
**Generated:** 2026-08-16T15:26:42.458694
**Model:** allenai/OLMoE-1B-7B-0924
**Device:** cuda:0

## 1. Exp3C Checkpoint Oracle Distance Matrices

| Checkpoint | Layer | Pairs | CSV Hash (first 12) |
|---|---|---|---|
| checkpoint_10 | first | 384 | 23ad593c2950 |
| checkpoint_10 | middle | 384 | fd058a8c2fcb |
| checkpoint_10 | last | 384 | 0389308607ce |
| checkpoint_40 | first | 384 | 7ad733300f86 |
| checkpoint_40 | middle | 384 | 58a08a613e9e |
| checkpoint_40 | last | 384 | 00e122e0b465 |
| checkpoint_70 | first | 384 | ccf79630e0e0 |
| checkpoint_70 | middle | 384 | 48cd122fde32 |
| checkpoint_70 | last | 384 | 173c64b25c9e |
| checkpoint_100 | first | 2016 | 4f3bd7ec8949 |
| checkpoint_100 | middle | 2016 | 8a1d41593f26 |
| checkpoint_100 | last | 2016 | d99521a015c3 |

## 2. Exp3B q-Value Ranking

- Primary q: **4** (selected in Exp3B, fixed in Exp4)
- Secondary q: **6** (second-best across middle+last layers)
- Tertiary q: **3** (second-best for first layer)

## 3. Router Telemetry Availability

- Historical telemetry exists: **False**
- Can reconstruct from checkpoint + calibration: **True**
- Method: Forward pass of calibration dataset through each checkpoint revision. Router hooks capture logits, Top-k indices, probabilities, and input hidden states. This is highly structured and exactly reproducible because the calibration dataset is frozen (SHA256 verified) and inference is in eval mode with no dropout.

## 4. Calibration Dataset

- Path: `/home/sandlogic/LINGO/PROJECTS/Experiments-V3/experiments/experiment3c/data/calibration/calibration_3c_wikitext.pt`
- Exists: True
- SHA256 match: **True**

### Plots


*(Section extracted to adhere to format)*

### Output


*(Section extracted to adhere to format)*

### Conclusion


## Functional Displacement Magnitude Summary

### Layer: first

#### q4

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.002418 | 0.001279 | 0.006368 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.000745 | 0.000502 | 0.003360 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.003378 | 0.001950 | 0.010842 | 0 |

#### q6

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.001500 | 0.000742 | 0.003192 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.000742 | 0.000448 | 0.003302 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.003280 | 0.001492 | 0.007543 | 0 |

#### q3

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.002456 | 0.001295 | 0.007485 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.000709 | 0.000489 | 0.003172 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.003083 | 0.001792 | 0.007330 | 0 |

### Layer: middle

#### q4

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.001287 | 0.002096 | 0.016911 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.001550 | 0.001309 | 0.008392 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.002024 | 0.002823 | 0.021645 | 0 |

#### q6

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.001781 | 0.002051 | 0.016863 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.001558 | 0.001226 | 0.008328 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.002085 | 0.002691 | 0.021547 | 0 |

#### q3

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.001771 | 0.002181 | 0.016948 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.001655 | 0.001475 | 0.008283 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.002071 | 0.002802 | 0.021588 | 0 |

### Layer: last

#### q4

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.002929 | 0.002575 | 0.012873 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.003226 | 0.002603 | 0.009782 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.003434 | 0.002591 | 0.010076 | 0 |

#### q6

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.000763 | 0.000487 | 0.002581 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.000684 | 0.000473 | 0.002862 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.003255 | 0.002209 | 0.008496 | 0 |

#### q3

| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |
|---|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | 370,000 | 0.002474 | 0.002104 | 0.008118 | 0 |
| checkpoint_40 -> checkpoint_70 | 305,000 | 0.003187 | 0.003117 | 0.017061 | 0 |
| checkpoint_70 -> checkpoint_100 | 425,000 | 0.003556 | 0.003248 | 0.017705 | 0 |



---
# Experiment 6B — Predictive Models

## Layer: first

| Transition | Model | R² | RMSE | Cos Sim (Direction) |
|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | M0_Zero | 0.0000 | 0.001368 | 0.0000 |
| checkpoint_10 -> checkpoint_40 | M2_Exposure | -0.0004 | 0.001368 | -0.0642 |
| checkpoint_10 -> checkpoint_40 | M3_Pos_Exposure | -0.0004 | 0.001368 | -0.0642 |
| checkpoint_10 -> checkpoint_40 | M4_Interaction | -0.0020 | 0.001369 | -0.0815 |
| checkpoint_40 -> checkpoint_70 | M0_Zero | 0.0000 | 0.000449 | 0.0000 |
| checkpoint_40 -> checkpoint_70 | M2_Exposure | -0.0713 | 0.000465 | -0.0408 |
| checkpoint_40 -> checkpoint_70 | M3_Pos_Exposure | -0.0713 | 0.000465 | -0.0407 |
| checkpoint_40 -> checkpoint_70 | M4_Interaction | -0.0608 | 0.000463 | -0.0115 |
| checkpoint_70 -> checkpoint_100 | M0_Zero | 0.0000 | 0.001950 | 0.0000 |
| checkpoint_70 -> checkpoint_100 | M2_Exposure | -0.0278 | 0.001977 | -0.0508 |
| checkpoint_70 -> checkpoint_100 | M3_Pos_Exposure | -0.0278 | 0.001977 | -0.0506 |
| checkpoint_70 -> checkpoint_100 | M4_Interaction | -0.0283 | 0.001978 | -0.0425 |

## Layer: middle

| Transition | Model | R² | RMSE | Cos Sim (Direction) |
|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | M0_Zero | 0.0000 | 0.001230 | 0.0000 |
| checkpoint_10 -> checkpoint_40 | M2_Exposure | -0.0286 | 0.001247 | -0.2207 |
| checkpoint_10 -> checkpoint_40 | M3_Pos_Exposure | -0.0285 | 0.001247 | -0.2190 |
| checkpoint_10 -> checkpoint_40 | M4_Interaction | -0.0314 | 0.001249 | -0.2015 |
| checkpoint_40 -> checkpoint_70 | M0_Zero | 0.0000 | 0.001014 | 0.0000 |
| checkpoint_40 -> checkpoint_70 | M2_Exposure | -0.0725 | 0.001050 | -0.1603 |
| checkpoint_40 -> checkpoint_70 | M3_Pos_Exposure | -0.0725 | 0.001050 | -0.1602 |
| checkpoint_40 -> checkpoint_70 | M4_Interaction | -0.0698 | 0.001049 | -0.1224 |
| checkpoint_70 -> checkpoint_100 | M0_Zero | 0.0000 | 0.001737 | 0.0000 |
| checkpoint_70 -> checkpoint_100 | M2_Exposure | -0.0731 | 0.001799 | -0.2089 |
| checkpoint_70 -> checkpoint_100 | M3_Pos_Exposure | -0.0732 | 0.001799 | -0.2089 |
| checkpoint_70 -> checkpoint_100 | M4_Interaction | -0.0821 | 0.001807 | -0.1365 |

## Layer: last

| Transition | Model | R² | RMSE | Cos Sim (Direction) |
|---|---|---|---|---|
| checkpoint_10 -> checkpoint_40 | M0_Zero | 0.0000 | 0.001950 | 0.0000 |
| checkpoint_10 -> checkpoint_40 | M2_Exposure | -0.0398 | 0.001988 | -0.0572 |
| checkpoint_10 -> checkpoint_40 | M3_Pos_Exposure | -0.0397 | 0.001988 | -0.0569 |
| checkpoint_10 -> checkpoint_40 | M4_Interaction | -0.0440 | 0.001992 | -0.0608 |
| checkpoint_40 -> checkpoint_70 | M0_Zero | 0.0000 | 0.002073 | 0.0000 |
| checkpoint_40 -> checkpoint_70 | M2_Exposure | -0.0221 | 0.002095 | -0.1055 |
| checkpoint_40 -> checkpoint_70 | M3_Pos_Exposure | -0.0220 | 0.002095 | -0.1051 |
| checkpoint_40 -> checkpoint_70 | M4_Interaction | -0.0186 | 0.002092 | -0.0826 |
| checkpoint_70 -> checkpoint_100 | M0_Zero | 0.0000 | 0.002151 | 0.0000 |
| checkpoint_70 -> checkpoint_100 | M2_Exposure | -0.0339 | 0.002187 | -0.1577 |
| checkpoint_70 -> checkpoint_100 | M3_Pos_Exposure | -0.0339 | 0.002187 | -0.1574 |
| checkpoint_70 -> checkpoint_100 | M4_Interaction | -0.0283 | 0.002181 | -0.0866 |



---
# Experiment 6B — Advanced Analysis & Empirical Laws

## Q-Robustness
Primary Q: 4, Secondary Q: 6
Is Robust: True

## Empirical Evolution Law Candidates
| Layer | R2 M3 (Pos+Exp) | R2 M4 (Interaction) | Selected Law |
|---|---|---|---|
| first | -0.0332 | -0.0304 | No compact law found |
| middle | -0.0580 | -0.0611 | No compact law found |
| last | -0.0319 | -0.0303 | No compact law found |
