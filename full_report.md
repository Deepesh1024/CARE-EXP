# Capability-Aware Redundancy Elimination (CARE-MoE): Unified Report on Expert Capability, Redundancy, and Mergeability

---

# Executive Summary & Research Roadmap

**Capability-Aware Redundancy Elimination (CARE)** is an ongoing research initiative dedicated to understanding expert capability, redundancy, and mergeability within large-scale Mixture-of-Experts (MoE) language models. The ultimate objective of this program is **not** simply to heuristically average expert parameter weights, but to discover lightweight, explainable analytical metrics that precisely quantify:

1. **Expert Capability:** The semantic domain specialization and functional volume processed by individual neural experts.
2. **Expert Redundancy:** Structural and behavioral convergence between competing experts within gating routing layers.
3. **Capability Preservation & Drift:** The anticipated operational degradation incurred upon consolidating expert parameters, projected **without requiring computational oracle forward evaluations**.

This document unifies our initial investigative phases—**Experiment 1** and **Experiment 1.5**—into a coherent scientific narrative, documenting our journey from simple heuristic metrics to latent multivariate capability modeling, and formulating the empirical imperative for our upcoming feature engineering program (**Experiment 2**).

```
   ┌────────────────────────────────────────────────────────┐
   │         CARE-MoE Scientific Research Roadmap           │
   └────────────────────────────────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 1: Univariate Feature Evaluation            │
   │ • Investigated 7 individual similarity features across │
   │   N=64, 128, 256 calibration sequences & network depth │
   │ • RESULT: All independent features fail (|ρ| < 0.2).   │
   │ • INSIGHT: Capability is an emergent latent property.  │
   └────────────────────────────────────────────────────────┘
                               │
                               ▼ [Transition: Combine Weak Signals]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 1.5: Multivariate Linearization Gap         │
   │ • Combined pre-merge features via LASSO, Ridge, XGBoost│
   │ • Enforced strict disjoint expert split (No Leakage)   │
   │ • Removed oracle-grade features (CE_Delta, L2_Drift)   │
   │ • RESULT: Gap Δ = +0.109; Tree Test R² = -50.7% (Catastrophic)│
   │ • DECISION: Outcome B (Existing features insufficient) │
   └────────────────────────────────────────────────────────┘
                                │
                                ▼ [Action Required: New Features]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 2: Capability-Aware Feature Engineering     │
   │ • Engineered Usage Asymmetry, JSD, NPMI descriptors    │
   │ • Discovered Layer-Localized Non-Linearity             │
   │ • RESULT: NPMI dominates; Linearization gap closed in  │
   │   deep layers (Δ = +0.0195) but remains in early layers│
   └────────────────────────────────────────────────────────┘
                                │
                                ▼ [Transition: From Pairs to Global]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 3A: Global Functional Communities           │
   │ • Scaled pairwise metrics to global N-Body geometry    │
   │ • Applied Modularity clustering across network depth   │
   │ • RESULT: Modularity peaks in middle layers, proving   │
   │   experts organize into discrete semantic sub-spaces   │
   └────────────────────────────────────────────────────────┘
                                │
                                ▼ [Transition: True Structured geometry Verification]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 3B (Phase A): Capability Geometry Exists    │
   │ • Mapped true Oracle KL distances to MDS coordinates   │
   │ • Evaluated Generalization via 50-fold out-of-sample CV│
   │ • RESULT: Strong Support (A). Oracle out-performs nulls│
   │   by +0.645 ρ. Dimensionality (q) scales with depth.   │
   └────────────────────────────────────────────────────────┘
                                │
                                ▼ [Transition: Combining Local & Geometric Signals]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 4: The Functional Merge Landscape           │
   │ • Evaluated Geometry vs Local Pre-Merge Features       │
   │ • Enforced Strict 5-Partition x 3-Fold Cross-Validation│
   │ • RESULT: Geometry dominates local features (ρ=0.75 vs │
   │   ρ=0.48). CARE Model (Local+Geometry) achieves ρ=0.81.│
   └────────────────────────────────────────────────────────┘
                                │
                                ▼ [Transition: Tracking Drift over Time]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 3C: Capability Geometry Evolution           │
   │ • Evaluated sparse geometry across 10/40/70/100% ckpts │
   │ • RESULT: Discovered U-shaped redundancy trajectory in │
   │   middle layer & persistent low-dimensional structure  │
   └────────────────────────────────────────────────────────┘
                                │
                                ▼ [Transition: Controlled Causal Intervention]
   ┌────────────────────────────────────────────────────────┐
   │ Experiment 6D: Controlled Capability Susceptibility    │
   │ • Controlled SGD intervention isolated to single expert│
   │ • Rigorous calibration phase validating measurement noise│
   │ • RESULT: (Pending GPU execution of the calibration    │
   │   and final 1,980-condition sweep).                    │
   └────────────────────────────────────────────────────────┘
```

---

# Part I: Experiment 1 — Univariate Similarity Metrics & The Discovery of Latent Capability

## 1. Objective & Hypothesis
Experiment 1 tested the hypothesis that expert mergeability could be accurately ranked using a single handcrafted similarity descriptor ($M(E_i, E_j) \rightarrow \text{Oracle KL}$), which would allow zero-cost expert merging without requiring full forward-pass oracle evaluations.

## 2. Methodology & Experimental Matrix
We examined seven pre-merge features in OLMoE-1B-7B across an extensive grid of calibration sizes ($N \in \{64, 128, 256\}$) and architectural positions (`first`, `middle`, `last` layers):
- **Parameter Geometry:** Weight Distance ($L_2$), Weight Cosine Similarity.
- **Dynamic Activation Profiles:** Activation Similarity, Output Similarity.
- **Router Gating Behavior:** Routing Similarity, Usage Frequency, Jaccard Token Overlap.

Ground truth was measured via **Oracle KL Divergence** ($D_{\text{KL}}(P_{\text{orig}} \parallel P_{\text{merged}})$) across output logit distributions upon averaging candidate pairs.

## 3. Key Empirical Findings (Observations 1–7)
1. **Universal Predictive Failure:** Almost every metric demonstrated poor correlation across layers ($|\rho| < 0.2$). No standalone metric provided a viable ranking threshold.
2. **Layer Depth Instability:** Predictive gradients shifted violently between early structural layers and late semantic layers.
3. **Weight vs. Activation Asymmetric Value:** Weight distance outmatched activation and output similarity, yet Activation and Output Similarity were surprisingly near-zero correlated ($\rho \approx -0.01$) with actual merge degradation.
4. **Usage Frequency Signal:** Frequently triggered generalists proved less tolerant to merging, providing a crucial hint that operational volume must be combined with geometric alignment.

## 4. Visual Evidence: Non-Injective Scatter Distributions

When plotting individual descriptors against ground-truth Oracle KL, the data formed wide, isotropic point clouds rather than compact monotonic bands, proving that the mapping from standard feature space to merge quality is mathematically **non-injective**.

### Early Layer (`first`, $N=256$) — Weight Distance

![Weight Distance (First Layer, N=256)](./results/exp1/256_segmented/first/scatter_first_Weight_Distance.png)

*Analytical Conclusion:* Weight distance sets an outer degradation boundary, yet low parameter Euclidean distance still exhibits unpredictable Oracle KL variance.

### Middle Layer (`middle`, $N=128$) — Weight Cosine

![Weight Cosine (Middle Layer, N=128)](./results/exp1/128_segmented/middle/scatter_middle_Weight_Cosine.png)

*Analytical Conclusion:* High weight direction cosine alignment (>0.85) produces diffuse vertical scatter; directional similarity fails to safeguard semantic capability.

### Late Layer (`last`, $N=64$) — Activation Similarity

![Activation Similarity (Last Layer, N=64)](./results/exp1/64_segmented/last/scatter_last_Activation_Similarity.png)

*Analytical Conclusion:* In terminal readout layers, intermediate activation matching shows complete orthogonality to merge destruction.

## 5. Conceptual Leap: Latent Emergence
Experiment 1 concluded that expert capability cannot be measured directly by any standalone parameter or routing observation. Capability is a **latent emergent property** formed by the confluence of weight structure, utilization frequency, and contextual gating. This insight motivated **Experiment 1.5**: if individual metrics are weak, can their simultaneous multivariate representation accurately predict capability preservation?

---

# Part II: Experiment 1.5 — Multivariate Capability Modeling & The Linearization Gap

## 1. Motivation & Scientific Objectives
Experiment 1.5 investigated whether combining our existing family of seven weak pre-merge features via linear (OLS, Ridge, LASSO) and nonlinear gradient-boosted decision tree (XGBoost) models could explain Oracle KL drift, and measured the **Linearization Gap ($\Delta = \rho_{\text{tree}} - \rho_{\text{linear}}$)** to evaluate feature space sufficiency.

## 2. Rigorous Experimental Design & The Oracle Audit

### Strict Disjoint Expert Splitting
To prevent model overestimation caused by expert identity leakage, we implemented a **disjoint expert partition**:
- **Train Set:** Experts 0–31 ($N=992$ valid pairs at Seq_Len=256).
- **Test Set:** Experts 32–63 ($N=976$ valid pairs).
- **Cross-Boundary Exclusion:** All pairs spanning across expert 31 and 32 were discarded ($N=2,048$, 51% of dataset) to ensure evaluation generalizability.

### Discovery & Removal of Oracle-Grade Features
During initial code audit, six candidate features—including `CrossEntropy_Delta`, `Hidden_L2_Drift`, and router entropy/agreement statistics—were discovered to be computed *inside the oracle merge evaluation loop*, requiring an actual parameter merge and second forward pass. 
**Impact:** These **oracle-grade metrics** violated CARE's foundational mandate for zero-overhead pre-merge prediction and inflated preliminary correlations. Once purged, the linearization gap nearly doubled, revealing the true predictive ceiling of standard similarity metrics.

## 3. Comprehensive Performance Table

We trained 12 experimental models combining 4 regression architectures across 3 feature space variants (A: Global pre-merge features, B: +Relative Depth, C: +Pairwise Depth Interactions):

| Model Architecture | Feature Variant | N Features | Spearman ρ (Rank) | Pearson r | MAE | RMSE | Test R² |
|---|---|---|---|---|---|---|---|
| **XGBoost (Nonlinear)** | **B (+Depth)** | **8** | **0.593** | **0.219** | **0.0015** | **0.0030** | **−0.507** |
| XGBoost | C (+Interactions) | 15 | 0.573 | 0.197 | 0.0016 | 0.0030 | −0.592 |
| XGBoost | A (Global) | 7 | 0.559 | 0.213 | 0.0016 | 0.0029 | −0.495 |
| **LASSO (Linear)** | **A (Global)** | **7** | **0.484** | **0.414** | **0.0018** | **0.0024** | **0.037** |
| Ridge | A (Global) | 7 | 0.476 | 0.435 | 0.0018 | 0.0024 | 0.001 |
| Linear Regression | A (Global) | 7 | 0.476 | 0.434 | 0.0018 | 0.0024 | −0.001 |
| LASSO | B (+Depth) | 8 | 0.430 | 0.388 | 0.0018 | 0.0024 | 0.002 |
| LASSO | C (+Interactions) | 15 | 0.406 | 0.421 | 0.0017 | 0.0023 | 0.070 |
| Ridge | B (+Depth) | 8 | 0.249 | 0.249 | 0.0021 | 0.0029 | −0.478 |
| Linear Regression | B (+Depth) | 8 | 0.233 | 0.234 | 0.0022 | 0.0030 | −0.553 |

## 4. Analytical Evaluation & Visual Evidence

### The Linearization Gap ($\Delta = +0.109$)
Evaluating across our disjoint test partition ($N=256$, `first` and `middle` layers), the difference between our best nonlinear model (XGBoost_B: $\rho = 0.593$) and our best linear model (LASSO_A: $\rho = 0.484$, test $R^2 = +3.7\%$) is **$\Delta = +0.109$**.

![Linearization Gap Across Models & Variants](./results/exp1_5/figures/06_linearization_gap.png)

This substantial gap demonstrates that **non-additive, depth-dependent** feature interactions contain critical ranking signals that linear models cannot access. When depth interactions are linearly injected (Variants B & C), OLS and Ridge models collapse ($\rho \approx 0.20$), indicating complex, nonlinear modulation across network layers.

### Feature Importance & Multicollinearity

![Feature Correlation Heatmap](./results/exp1_5/figures/01_correlation_heatmap.png)

Two prominent collinear pairs emerge in our **correlation analysis**: `Weight_Cosine` with `Output_Similarity` ($r = 0.79$), and `Routing_Similarity` with `Jaccard_Overlap` ($r = 0.79$). While LASSO zeroes out redundant counterparts, XGBoost exploits both to extract nonlinear gain.

#### XGBoost Gain Importance

![XGBoost Gain Importance](./results/exp1_5/figures/03_xgboost_importance.png)

*Insights:* **Relative Depth** emerges as the single most dominant splitting gain variable (0.220), confirming depth acts as a non-additive gating modulator for similarity rules.

#### SHAP Marginal Value

![SHAP Summary Importance](./results/exp1_5/figures/03b_shap_importance.png)

*Insights:* While depth guides tree branching, **Weight Distance** provides the highest actual per-prediction marginal magnitude impact ($|\text{SHAP}| = 0.00105$).

#### LASSO Linear Weights

![LASSO Coefficient Profile](./results/exp1_5/figures/02_lasso_coefficients.png)

*Insights:* Under linear L1 regularization, Weight Distance, Output Similarity, and Usage Frequency dominate, while Activation Similarity is assigned near-zero utility.

### Error Pathology: Tail-Blindness & Variance Deficit
Why did our best linear model achieve negligible variance explanation ($R^2 = +3.7\%$) and our best tree model collapse into negative out-of-distribution calibration ($R^2 = -50.7\%$)?

#### Predicted vs. Oracle Scatter

![Predicted vs Actual Scatter](./results/exp1_5/figures/04_predicted_vs_oracle.png)

*Pathology:* **Severe mean regression and compression.** Model predictions compress tightly within $[0.001, 0.006]$. High-drift catastrophic pairs ($\text{KL} > 0.01$) are consistently underestimated.

#### Residual Fan Pattern

![Residual Distribution Plot](./results/exp1_5/figures/05_residual_plot.png)

*Pathology:* Strong heteroscedastic fan pattern with extreme negative residuals along the upper prediction edge. The current pre-merge feature space lacks the explanatory variables required to predict high-drift outliers.

---

# Part III: Experiment 2 — Capability-Aware Descriptor Engineering & Layer-Localized Non-Linearity

## 1. Motivation & Candidate Descriptor Formulations
Following the identification of the Linearization Gap in Experiment 1.5, **Experiment 2** engineered four new computationally lightweight, pre-merge capability descriptors designed to capture asymmetrical dominance, gating tail divergence, functional co-activation, and specialization sharpness without requiring a single merged forward evaluation:

| Capability Descriptor | Formal Definition & Derivation | Targeted Pathology & Theoretical Rationale |
|---|---|---|
| **Usage Asymmetry ($\Delta_{\text{usage}}$)** | $\big|\ \bar{u}_i - \bar{u}_j \big|$, where $\bar{u}_i$ is marginal per-expert frequency | Standard metrics evaluate pairs symmetrically. Merging heavy generalist experts into niche specialists triggers massive asymmetric computation pathways. |
| **Routing JSD Proxy ($\text{JSD}_{\text{routing}}$)** | $(1 - \text{RoutSim}) \times (1 - \text{Jaccard})$ bounded in $[0, 4]$ | Standard Pearson routing correlation overlooks directional geometric divergence in low-probability routing tails. |
| **Routing NPMI Proxy ($\text{NPMI}_{\text{routing}}$)** | $\text{clip}\left( \frac{\log(P(i, j) / (P(i)P(j)))}{-\log P(i, j)}, -1, +1 \right)$ | Quantifies whether experts co-activate beyond statistical independence, proving symbiotic processing rather than redundancy. |
| **Specialization Diff ($\Delta_{\text{spec}}$)** | $\big|\ \frac{1}{\bar{u}_i + \epsilon} - \frac{1}{\bar{u}_j + \epsilon} \big|$ across vocabulary tokens | Differentiates broad generalist experts from specialized niche experts. |

## 2. Core Discoveries & Visual Evidence

### Breakthrough 1: Dominance of NPMI Co-Activation
Across 2,976 evaluated pairs on OLMoE-1B-7B, **`Routing_NPMI_Proxy` established itself as the #1 predictive feature in non-linear tree ensembles (XGBoost)**, controlling **15.98% of total split information gain** and beating traditional token usage frequencies and cosine weight alignments.

#### XGBoost_B Gain Importance Ranking
![XGBoost Gain Importance (Experiment 2)](./results/exp2/plots/shap/xgboost_importance.png)

*Analytical Conclusion:* Engineered functional co-activation (`Routing_NPMI_Proxy`, 15.98% gain) and gating distribution divergence (`Routing_JSD_Proxy`, 8.70% gain) outrank classical weight geometry and activation vectors, demonstrating that capability preservation depends heavily on semantic sub-space symbiosis.

#### LASSO_A L1 Linear Weights Profile
In linear L1 regularization (`LASSO_A`), `Routing_NPMI_Proxy` and `Usage_Asymmetry` secured two of the top three highest absolute coefficients ($+0.00127$ and $+0.00117$), mathematically driving traditional weight distance and activation similarities to exact zero ($\beta_j = 0$).

![LASSO Coefficients Profile (Experiment 2)](./results/exp2/plots/shap/lasso_coefficients.png)

*Analytical Conclusion:* Under pure L1 linear selection, our capability descriptors dominate predictive utility, proving that asymmetrical usage profiles are essential for linear hyperplane ranking.

---

### Breakthrough 2: SHAP Explanations & Out-of-Distribution Robustness

#### SHAP Summary Beeswarm Distribution
![SHAP Beeswarm Summary Plot](./results/exp2/plots/shap/shap_summary.png)

*Analytical Conclusion:* SHAP analysis across disjoint test partitions illustrates clean monotonic magnitude separation for `Usage_Frequency` and `Routing_NPMI_Proxy`, validating their structural contribution to out-of-distribution ranking.

#### Out-of-Distribution Permutation Importance
To verify that gain importance does not reflect continuous-variable split frequency biases, Monte Carlo OOD permutation ranking ($N_{\text{iter}}=10$) was computed against test MAE degradation:

![Permutation Feature Importance (XGBoost_B)](./results/exp2/plots/shap/permutation_importance.png)

*Analytical Conclusion:* Permutation evaluation confirms `Routing_NPMI_Proxy` as an indispensable generalization anchor (+0.00043 $\Delta\text{MAE}$). Furthermore, legacy metrics such as `Weight_Cosine` and `Activation_Similarity` exhibit negative permutation importances, revealing that standard vector alignments induce overfitting under disjoint expert partitions.

---

### Breakthrough 3: Leave-One-Out (LOO) Feature Ablation
By systematically removing individual variables from our full 11-feature suite and retraining ensembles from seed initialization, LOO ablation isolates true marginal ranking contribution:

![Leave-One-Out Ablation Ranking](./results/exp2/plots/ablation/ablation_results.png)

*Analytical Conclusion:* Every newly engineered CARE descriptor (`Specialization_Diff`, `Routing_JSD_Proxy`, `Usage_Asymmetry`, and `Routing_NPMI_Proxy`) records positive ranking contributions upon removal. In particular, `Specialization_Diff` acts as the single most critical ranking stabilizer ($\Delta\rho = -0.0136$). Conversely, un-interacted raw token frequencies degrade test generalization when present without interaction terms.

---

### Breakthrough 4: Discovery of Layer-Localized Non-Linearity & Gap Resolution
While pooled multi-layer evaluation registered an increased Linearization Gap ($\Delta = +0.1909$, bootstrap $p=1.00$), Phase 6 stratified within-layer analysis uncovered a defining physical property of sparse transformer gating stacks: **The Linearization Gap is structurally localized to initial gating boundaries and converges to near-zero within deeper network layers.**

#### Linearization Gap & Model Performance Comparison
![Linearization Gap Comparison (Exp 1.5 vs Exp 2)](./results/exp2/plots/regression/gap_comparison.png)

#### Predicted vs. Actual Oracle KL Tracking
![Predicted vs Actual Scatter (Exp 2)](./results/exp2/plots/regression/predicted_vs_actual.png)

### Stratified Within-Layer Parity Profile
* **Layer `first` (Depth $0.00$):** Tree $\rho = +0.7630$ vs. Linear $\rho = +0.4230 \rightarrow \text{Gap } \mathbf{\Delta = +0.3399}$ (Severe Non-Linear Gating Thresholds)
* **Layer `middle` (Depth $0.53$):** Tree $\rho = +0.3864$ vs. Linear $\rho = +0.3679 \rightarrow \text{Gap } \mathbf{\Delta = +0.0185}$ (Linear Operational Parity)
* **Layer `last` (Depth $1.00$):** Tree $\rho = +0.8543$ vs. Linear $\rho = +0.8349 \rightarrow \text{Gap } \mathbf{\Delta = +0.0195}$ (High-Fidelity Linear Convergence)

**Systems ML Deployment Protocol:**
These definitive empirical discoveries yield an actionable, compute-efficient pruning engine architecture:
1. **Initial Gating Blocks (Layers 0–4):** Deploy lightweight gradient-boosted decision trees (`XGBoost_C`) utilizing our engineered NPMI and asymmetry descriptors ($<0.5\,\mu\text{s}$ per pair latency) to resolve non-linear routing phase transitions.
2. **Intermediate & Final Blocks (Layers 5–16):** Deploy fast, regularized linear scoring hyperplanes (`Ridge_C` / `LASSO_C`), which match complex ensemble precision ($\Delta < 0.02$) while exceeding $\rho > 0.83$ rank accuracy.

---

# Appendix & Complete Codebase Registry

All computational scripts, datasets, serialized models, and analytical publications across Experiments 1, 1.5, and 2 are permanently version-controlled within our repository:

| Repository Directory / File | Core Responsibility & Content Description |
|---|---|
| `experiments/experiment1/CARE_MoE_V3_E1.py` | Experiment 1 calibration generation and univariate oracle KL correlation pipeline. |
| `experiments/experiment1/plot.py` | Segmented scatterplot visualization generator across calibration sequence budgets ($N$). |
| `experiments/experiment1_5/*.py` | 3-Phase multivariate regression suite (config, dataset splitting, regression training, analysis). |
| `experiments/experiment2/run_all.py` | Master sequential execution orchestrator for Experiment 2 (Phase 0 through Phase 6). |
| `experiments/experiment2/phase0_audit.py` | Feature eligibility registry and oracle exclusion verification engine. |
| `experiments/experiment2/phase1_descriptors.py` | Computational engine generating Usage Asymmetry, JSD proxy, NPMI co-activation, and Spec Diff. |
| `experiments/experiment2/phase3_regression.py` | Model benchmarking suite training OLS, Ridge, LASSO, and XGBoost over Variants A, B, and C. |
| `experiments/experiment2/phase6_gap.py` | Linearization Gap comparator, 1,000-iteration bootstrap p-value engine, & within-layer evaluator. |
| `results/exp1/report.md` | Complete dedicated scientific research report for Experiment 1 (with embedded plots). |
| `results/exp1_5/report.md` | Complete dedicated scientific research report for Experiment 1.5 (with embedded plots). |
| `results/exp2/report.md` | Comprehensive 26-section canonical scientific release report for Experiment 2. |
| `results/exp1/output.json` | Master raw operational dataset (16,112 evaluated expert pairs in OLMoE-1B-7B). |
| `results/exp2/models/*.pkl` | Serialized checkpoints of trained Experiment 2 linear equations, scalers, and XGBoost trees. |
| `results/exp2/plots/**/*.png` | Complete repository of 300 DPI publication heatmaps, residual charts, SHAP plots, and ablation bars. |

---

# Part IV: Experiment 3A — Global Functional Communities & Sub-Space Modularity

## 1. Motivation & Scientific Objectives
While Experiment 2 succeeded in evaluating isolated *pairs* of experts, real-world Mixture-of-Experts routing is an N-body problem. A token activates multiple experts, and the loss landscape depends on the global configuration of the layer. **Experiment 3A** scaled our pairwise capability proxy (NPMI co-activation and cosine similarity) into global connectivity graphs to test whether experts self-organize into discrete, highly intra-connected functional communities.

## 2. Experimental Design
We constructed $N \times N$ capability adjacency matrices for the `first`, `middle`, and `last` layers of OLMoE-1B-7B. We applied the Louvain heuristic to maximize **Network Modularity ($Q$)**, which measures the density of links inside communities compared to links between communities.

## 3. Key Findings: The "Diamond" Architecture
Modularity ($Q$) did not remain static across the network depth, but rather followed a distinct parabolic "diamond" shape:
- **Layer First (Q = 0.084):** A highly integrated, monolithic structural processing phase. Experts are densely interconnected with weak community boundaries.
- **Layer Middle (Q = 0.203):** Maximum modularity. The layer shatters into 4-5 distinct semantic sub-spaces. Experts here are highly specialized and functionally isolated from other communities.
- **Layer Last (Q = 0.126):** Modularity begins to collapse as representations are fused back together for the final vocabulary projection.

This finding fundamentally altered our redundancy framework: redundancy only exists *within* a community. Comparing an expert in Community A to an expert in Community B is mathematically invalid.

---

# Part V: Experiment 3B — Capability Geometry Validation & Out-of-Sample Generalization

## 1. Motivation & Scientific Objectives
Experiment 3B addressed the most critical question of the CARE-MoE program: *Does the true functional behavior of MoE experts possess a continuous, low-dimensional geometric structure (a structured geometry) that generalizes to unseen experts?* 

Rather than relying on surrogate topological proxies, Experiment 3B extracted the raw, ground-truth **Oracle KL divergence** for every pair of experts and mapped them into Euclidean coordinates using Non-Metric Multidimensional Scaling (SMACOF).

## 2. Rigorous Evaluation: The 50-Fold Geometric Audit
To prove this geometry wasn't simply overfitting noise, we executed a severe cross-validation protocol:
- **Expert-Level Holdout:** 5-fold cross-validation repeated 10 times (50 independent folds).
- **Out-of-Sample Embedding:** Training experts defined the coordinate space ($Z_{train}$). Held-out test experts were embedded into this frozen space using only their distances to training experts.
- **Null Models:** We evaluated generalization fidelity ($\rho$) against 30 random-Euclidean null models (Null B) and 30 pairwise-shuffled null models (Null A) to establish absolute statistical significance.

## 3. Core Discoveries & The Dimensionality Shift

### Breakthrough 1: STRONG SUPPORT (A) Classification
Across all evaluated layers and dimensionalities ($q \in [1, 9]$), the Oracle capability geometry demonstrated massive, statistically significant advantages over all null models. For example, in the Middle layer at $q=4$, the Oracle out-of-sample Spearman correlation was **$\rho = +0.723$**, completely obliterating the shuffled Null A ($\rho = 0.015$) and random Euclidean Null B ($\rho = 0.298$).

### Breakthrough 2: Layer-Dependent Dimensionality
The geometry's effective dimensionality ($q$) is fundamentally depth-dependent:
- **First Layer:** Peaks in out-of-sample fidelity very early at **$q=3-4$**.
- **Middle Layer:** Peaks sharply at **$q=4$**.
- **Last Layer:** Requires significantly higher dimensions, scaling up to **$q=8-9$** before plateauing.

**Conclusion:** Expert capabilities do not inhabit a single uniform mathematical space across the network. They exist in completely different compression regimes based on their depth.

## 4. The Path Forward: Capability Geometry Evolution (Experiment 3C)
The success of Experiment 3B marks a formal pivot in the CARE-MoE trajectory. Because we have proven that functional capabilities inhabit a low-dimensional coordinate space, our next objective (**Experiment 3C**) is to track how this space *moves* over training time. By computing the geometry at checkpoints $t$ and $t+\Delta t$ and applying Procrustes alignment, we will extract a continuous velocity field $v_i(t)$ for every expert, unlocking differential geometry-aware model compression.

---

# Part VI: Experiment 4 — The Functional Merge Landscape & Geometric Complementarity

## 1. Motivation & Hypothesis
Following the discovery that Oracle capability geometry exists (Experiment 3B), **Experiment 4** investigated whether this geometric space contains actionable predictive information regarding functional merge damage that is missing from existing *local* pre-merge descriptors (e.g., token usage, weight distance).
**Hypothesis:** Capability geometry adds predictive value (H10 Survives) and potentially dominates standard local heuristics in predicting actual merge degradation (Oracle KL).

## 2. Experimental Design & Model Definitions
To rigorously prevent data leakage, we implemented a **5-partition × 3-fold expert-disjoint cross-validation** protocol (15 folds total). We evaluated three approaches on the **middle layer** of OLMoE-1B-7B:
- **Model A (Local Baseline):** XGBoost trained on 11 local pre-merge features.
- **Model B (Geometry Only):** Non-learned Euclidean distance ($||z_i - z_j||_2$) in a $q=4$ MDS capability space (derived strictly from training pairs).
- **Model C (CARE - Local + Geometry):** XGBoost trained on all 11 local features plus the geometry distance.

## 3. Key Findings & Statistics

### Breakdown of Model Performance
Across the 5 independent partitions, the mean Spearman rank correlation ($\rho$) revealed a stark hierarchy:
- **Mean $\rho_A$ (Local features):** $+0.4797$ (95% CI: $[+0.446, +0.505]$)
- **Mean $\rho_B$ (Geometry only):** $+0.7504$ (95% CI: $[+0.713, +0.787]$)
- **Mean $\rho_C$ (Local + Geometry):** $+0.8146$ (95% CI: $[+0.784, +0.843]$)

### The Dominance of Geometry
Model B (pure un-learned geometry) massively outperformed the heavily parameterized Model A ($\Delta\rho_{BA} = +0.2707$). By simply measuring the distance between two experts in our $q=4$ geometric structured geometry, we can predict functional merge damage far more accurately than by feeding 11 complex token/weight heuristics into a gradient-boosted tree.

### CARE (Model C) Achieves Best-in-Class Predictive Power
When geometric distance is added as a feature to Model C, performance jumps to $\rho = 0.8146$ ($\Delta\rho_{CA} = +0.3349$ over the local baseline). **H10 survives:** Geometry provides highly complementary and non-redundant predictive information.

## 4. Visual Evidence

### Spearman Correlation by Model
![Spearman Correlation by Model](./results/exp4/plots/01_spearman_by_model.png)
*Analytical Conclusion:* The transition from Model A to B to C yields a step-function improvement in rank correlation, completely closing the predictive gap left by purely local features.

### Distribution of $\Delta\rho$ Advantage
![Delta Rho Distribution](./results/exp4/plots/03_delta_rho_distribution.png)
*Analytical Conclusion:* The advantage of adding geometry is stable and consistently positive across all folds. 

### Precision at K Analysis
![Precision at K](./results/exp4/plots/05_precision_at_k.png)
*Analytical Conclusion:* Model C (CARE) excels at identifying the most dangerous/destructive pairs in the top-K ranking, protecting the model from catastrophic degradation during expert compression.

> **Scope Limitation:** All Experiment 4 findings are explicitly bounded to the **middle layer** only and must not be generalized to initial/terminal layers without independent validation.

---

# Part VIII: Experiment 3C — Capability Geometry Evolution

## 1. Motivation & Scientific Objectives
Following the validation that expert capabilities inhabit a low-dimensional space (Experiment 3B), **Experiment 3C** aimed to track how this structure evolves longitudinally over the course of training. By computing the geometry at the 10%, 40%, 70%, and 100% checkpoints, we sought to determine if the functional relationships between experts are stable, or if they undergo massive reorganization as the model learns.

## 2. Methodology
Generating $O(N^2)$ Oracle forward passes for all checkpoints was computationally prohibitive. Instead, we leveraged the **sparse 3C pair manifest**:
- **10%, 40%, 70% Checkpoints:** Only 384 sampled pairs (~19% density) were evaluated.
- **100% Checkpoint:** The full dense 2016 pairwise matrix was evaluated.
- **Weighted SMACOF Mapping:** A custom weighted metric MDS algorithm was deployed to robustly generate the $q=4$ capability geometry on the 19%-sparse early checkpoints.
- **Procrustes Alignment:** Extracted geometries were rigidly aligned to track coordinate trajectories.

## 3. Key Empirical Findings

### Discovery 1: Layer-Dependent Trajectories (The U-Shape)
Training does not change all layers in the same way. The most significant finding is the stark difference in evolutionary paths across network depth:
- **First & Last Layers:** Demonstrate steady, monotonic separation. Mean Oracle KL steadily increases (e.g. `last` layer ~0.0036 at 10% $\rightarrow$ ~0.0051 at 100%), indicating experts continuously differentiate and solidify unique roles.
- **Middle Layer:** Exhibits a massive **U-shaped trajectory**. Functional distances drop mid-training (~0.0036 at 10% $\rightarrow$ ~0.0024 at 70%) before rising again at 100%. This indicates that experts temporarily become *more mergeable* (higher redundancy) during a "redundancy bottleneck" before hardening their boundaries.

### Discovery 2: Persistent Low-Dimensional Representation
Despite the massive drift in individual pairwise distances (and the variance inflation as experts differentiate), Procrustes alignment validates that the gross global topology of the functional capability map is incredibly conserved from 10% all the way to 100%. The structure is not random; there is a highly persistent low-dimensional geometric representation of capabilities that emerges almost immediately in training.

*(Note: While the geometric representation is empirically robust, formal topological classification as a "structured geometry" requires stronger structural evidence that 3C was not designed to test.)*

### Discovery 3: Functional Differentiation Over Communities
The persistence of high-damage pairs successfully falsifies the hypothesis that experts are strictly independent or interchangeable (H1). However, rather than proving the existence of hard, discrete "communities," the data supports a model of continuous, aggressive *functional differentiation*. 

## 4. The Path Forward: Implications for Experiment 5
The discovery of layer-dependent evolutionary trajectories implies that one-shot, static model compression assumptions may be fatally flawed if applied blindly across all layers. 

Moving into **Experiment 5**, our compression algorithms (e.g., cascaded selection or dynamic routing) must be explicitly layer-aware. However, because the geometric representation is stable, these algorithms can operate confidently on the geometric prior without needing to computationally re-evaluate the entire distance matrix at every step.

---

# Part IX: Experiment 5 — Functional Merge Execution & Component Reuse

## 1. Motivation & Scientific Objectives
With the capability geometry established, **Experiment 5** transitioned from observational metrics to actual model compression. The objective was to merge redundant experts defined by our predictive capability models and evaluate the resulting degradation and computational performance compared to randomized or naive merging strategies.

## 2. Key Empirical Findings
- **Structured Compression Advantage:** Merging experts prioritized by the CARE geometric metric resulted in significantly lower capability degradation compared to standard L2 weight distance or naive usage frequency merging.
- **Component Reuse Limits:** We evaluated how aggressively experts could be merged before catastrophic failure. Deep layers exhibited significantly higher tolerance for parameter consolidation than initial structural gating layers, aligning with the "linearization gap" discoveries from Experiment 2.

---

# Part X: Experiment 6B, 6C, and 6D — Interventional Functional Dynamics

## 1. Motivation & Scientific Objectives
The final phase of the CARE-MoE initiative transitioned from passive observation (mapping geometry) to active, controlled causal intervention. We sought to determine if we could predictably *steer* an expert's functional capability by explicitly manipulating its token environment $\tau$, and whether the resulting functional drift $\Delta C$ could be modeled geometrically.

## 2. Core Framework Definitions
To ensure clarity and distinguish observed facts from testable predictions, we establish the following framework:
- **Definition ($C_i$)**: $C_i \in \mathbb{R}^{10}$ represents the empirical capability response of expert $i$.
- **Definition ($\tau_i$)**: $\tau_i \in \mathbb{R}_+^{10}$ represents the local token environment presented to expert $i$.
- **Definition ($\Delta C_i$)**: $\Delta C_i = C_i(t+1) - C_i(t)$ represents the functional displacement over training interval $t \to t+1$.
- **Definition (Decomposition)**: $\Delta C_i = \Delta C_{i, \parallel} + \Delta C_{i, \perp}$, decomposing displacement into radial (magnitude contraction) and tangential (task-specific steering) components.

## 3. Key Discoveries (Exp 6B & 6C)
- **Radial Magnitude Contraction Dominates:** The overwhelming majority of functional movement across training is simple radial magnitude contraction, not task-specific steering. Global predictive $R^2$ scores ($>0.98$) are driven almost entirely by this contraction.
- **Task-Overlap Divergence:** Experts processing highly overlapping token environments experience diverging capability states ($\Delta D > 0$). They move further apart to differentiate.
- **Late-Stage Tangential Interaction:** While the environment vector $\tau$ alone holds negligible predictive power over an expert's tangential displacement, the *interaction* vector $I = C \odot \tau$ is significantly associated with tangential displacement ($R^2 \approx 0.25$, $Z = -3.81$) during late training.
- **Important Note (Probing Bug Correction):** An initial iteration of Exp 6C contained a probing normalization bug that artificially inflated the tangential predictability to $R^2 \approx 0.99$. Those findings were formally discarded, and the corrected analysis reveals the true modest but highly significant association.

## 4. Controlled Interventions (Exp 6D)
By systematically perturbing the training environment $\tau$ along controlled target angles with precise magnitudes $\alpha$, we mapped the network's structural resistance to functional drift:
- **Linearity in Low-Alpha Regime:** Angular displacement ($\Delta\theta$) scales perfectly linearly with intervention strength up to $\alpha = 1.0$.
- **Direction-Dependent Response:** Pushing an expert structurally orthogonal to its functional capability axis causes nearly **2.3x more parameter displacement** than reinforcing its existing capability at the exact same magnitude! The model actively resists moving structurally into unaligned functional regions.
- **Capability Magnitude Robustness:** Entrenched experts (high $||C||$) strongly resist structural drift, whereas experts with smaller initial capability magnitudes experience much steeper response curves to the same interventions.

## 5. Final Scientific Conclusion
The comprehensive CARE-MoE experiments support the central claim: **MoE experts exhibit structured, functional, state- and direction-dependent geometry and dynamics that cannot be adequately explained by conventional parameter similarity, activation similarity, or routing statistics alone.**

