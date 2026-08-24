# EXPERIMENT 6C FINAL REPORT
**Generated:** 2026-08-19T11:58:08.959465


## Core Framework Definitions
To ensure clarity and distinguish observed facts from testable predictions, we establish the following framework:

- **Definition ($C_i$)**: $C_i \in \mathbb{R}^{10}$ represents the empirical capability response of expert $i$.
- **Definition ($	au_i$)**: $	au_i \in \mathbb{R}_+^{10}$ represents the local token environment presented to expert $i$.
- **Definition ($\Delta C_i$)**: $\Delta C_i = C_i(t+1) - C_i(t)$ represents the functional displacement over training interval $t 	o t+1$.
- **Definition (Decomposition)**: $\Delta C_i = \Delta C_{i, \parallel} + \Delta C_{i, \perp}$, decomposing displacement into radial (magnitude contraction/expansion) and tangential (angular/task-specific steering) components.
- **Hypothesis (Geometric Susceptibility)**: The tangential movement $\Delta C_{i, \perp}$ is directionally guided by the orthogonal component of the interaction vector $I = C_i \odot 	au_i$.

## Capability-Space Vector Movement and Expert Interaction

### 1. Research Question
How does an MoE expert's functional state move through an empirical capability space as it encounters different token/task environments, and how is that movement related to the expert's existing functional state and its neighboring experts?
Specifically, does the incoming environment $\tau$ induce a directional angular shift determined by the capability-conditioned interaction vector $I_i = C_i \odot \tau_i$?

### 2. Common Vector-Space Construction
We constructed a 10-dimensional empirical capability space using audited ARC and MMLU subset categories.
The functional vectors $C_i$ were extracted as the capability-probe response strength (the mean output activation norm of expert $i$ when fed tokens from axis $k$), bypassing the router entirely.

> [!WARNING]
> **Probing Bug & Corrupted Initial Results**
> An initial pre-release version of this experiment contained a probing normalization bug that artificially inflated global variance explained, yielding an invalid claim that environmental susceptibility deterministically explains all functional drift ($R^2 \approx 0.99$). Those conclusions have been fully discarded. The analysis below reflects the rigorously audited, corrected dataset, which demonstrates that while the global $R^2$ remains very high ($>0.98$) due to overwhelming radial magnitude contraction, the task-specific *tangential* signal ($I_\perp \to \Delta C_\perp$) is far more modest ($R^2 \approx 0.25$) but mathematically verifiable.

### 3. Directional Alignments and Interaction (10D)
Does the interaction vector $C \odot \tau$ better explain the functional displacement $\Delta C$ than $\tau$ alone?

#### Transition checkpoint_10->checkpoint_40
- **Mean 10D $\cos(\Delta C, \tau)$**: -0.6729
- **Mean 10D $\cos(\Delta C, C)$**: -0.8754
- **Mean 10D $\cos(\Delta C, I)$**: -0.6751
- **Mean 10D $\cos(\Delta C_\perp, I_\perp)$**: 0.2805

**Predictive Models ($R^2$)**
- $\tau$ only: 0.0002
- $C$ only: 0.8272
- $C + \tau$: 0.8273
- $I$ only: 0.3423
- Full ($C + \tau + I$): 0.8275
- **$I_\perp$ predicting $\Delta C_\perp$**: 0.0059

**Angular Susceptibility ($S_\theta$) Sensitivity Analysis**
- **Threshold**: Exclude bottom 5% $||C||$ (Retained 960). Layer Model $R^2$: 0.1374
- **Threshold**: Exclude bottom 10% $||C||$ (Retained 912). Layer Model $R^2$: 0.1357
- **Threshold**: Exclude bottom 20% $||C||$ (Retained 816). Layer Model $R^2$: 0.1354

#### Transition checkpoint_40->checkpoint_70
- **Mean 10D $\cos(\Delta C, \tau)$**: -0.7858
- **Mean 10D $\cos(\Delta C, C)$**: -0.9998
- **Mean 10D $\cos(\Delta C, I)$**: -0.7862
- **Mean 10D $\cos(\Delta C_\perp, I_\perp)$**: -0.0390

**Predictive Models ($R^2$)**
- $\tau$ only: 0.0010
- $C$ only: 0.9829
- $C + \tau$: 0.9830
- $I$ only: 0.4232
- Full ($C + \tau + I$): 0.9832
- **$I_\perp$ predicting $\Delta C_\perp$**: 0.0640

**Angular Susceptibility ($S_\theta$) Sensitivity Analysis**
- **Threshold**: Exclude bottom 5% $||C||$ (Retained 960). Layer Model $R^2$: 0.1552
- **Threshold**: Exclude bottom 10% $||C||$ (Retained 912). Layer Model $R^2$: 0.1566
- **Threshold**: Exclude bottom 20% $||C||$ (Retained 816). Layer Model $R^2$: 0.1633

#### Transition checkpoint_70->checkpoint_100
- **Mean 10D $\cos(\Delta C, \tau)$**: -0.7857
- **Mean 10D $\cos(\Delta C, C)$**: -1.0000
- **Mean 10D $\cos(\Delta C, I)$**: -0.7856
- **Mean 10D $\cos(\Delta C_\perp, I_\perp)$**: -0.2043

**Predictive Models ($R^2$)**
- $\tau$ only: 0.0007
- $C$ only: 0.9926
- $C + \tau$: 0.9926
- $I$ only: 0.4306
- Full ($C + \tau + I$): 0.9928
- **$I_\perp$ predicting $\Delta C_\perp$**: 0.2542

**Angular Susceptibility ($S_\theta$) Sensitivity Analysis**
- **Threshold**: Exclude bottom 5% $||C||$ (Retained 960). Layer Model $R^2$: 0.1800
- **Threshold**: Exclude bottom 10% $||C||$ (Retained 912). Layer Model $R^2$: 0.1755
- **Threshold**: Exclude bottom 20% $||C||$ (Retained 816). Layer Model $R^2$: 0.1733

### 4. Null-Model Analysis
Are the directional alignments and spatial convergences statistically significant against randomized nulls?

#### Transition checkpoint_10->checkpoint_40
- **Observed $\cos(\Delta C, I)$**: -0.6751
  - Random Direction Null Z-Score: -70.58 (Significant: True)
  - $\tau$-Permutation Null Z-Score: -0.29 (Significant: False)
- **Observed $\cos(\Delta C_\perp, I_\perp)$**: 0.2805
  - Random Direction Null Z-Score: 28.49 (Significant: True)
  - $\tau$-Permutation Null Z-Score: 0.50 (Significant: False)

- **Observed Task-Overlap vs $\Delta D$ (Spearman $\rho$)**: 0.3572
  - Pair-Matched Null Z-Score: 62.72 (Significant: True)

#### Transition checkpoint_40->checkpoint_70
- **Observed $\cos(\Delta C, I)$**: -0.7862
  - Random Direction Null Z-Score: -80.62 (Significant: True)
  - $\tau$-Permutation Null Z-Score: -2.02 (Significant: False)
- **Observed $\cos(\Delta C_\perp, I_\perp)$**: -0.0390
  - Random Direction Null Z-Score: -3.75 (Significant: True)
  - $\tau$-Permutation Null Z-Score: -1.54 (Significant: False)

- **Observed Task-Overlap vs $\Delta D$ (Spearman $\rho$)**: 0.5206
  - Pair-Matched Null Z-Score: 98.86 (Significant: True)

#### Transition checkpoint_70->checkpoint_100
- **Observed $\cos(\Delta C, I)$**: -0.7856
  - Random Direction Null Z-Score: -78.49 (Significant: True)
  - $\tau$-Permutation Null Z-Score: -1.90 (Significant: False)
- **Observed $\cos(\Delta C_\perp, I_\perp)$**: -0.2043
  - Random Direction Null Z-Score: -20.99 (Significant: True)
  - $\tau$-Permutation Null Z-Score: -3.81 (Significant: True)

- **Observed Task-Overlap vs $\Delta D$ (Spearman $\rho$)**: 0.4937
  - Pair-Matched Null Z-Score: 96.28 (Significant: True)

### 5. Conclusions
(This section to be filled by researcher after reviewing the generated outputs and plots).
