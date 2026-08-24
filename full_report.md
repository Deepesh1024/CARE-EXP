# Capability-Aware Redundancy Elimination (CARE-MoE): Unified Report

---

## Experiment 1: Univariate Similarity Metrics

### Hypothesis
Expert mergeability can be accurately ranked using a single handcrafted similarity descriptor (e.g., Weight Distance, Activation Similarity), allowing zero-cost expert merging without requiring full forward-pass oracle evaluations.

### Experiment
- **Dataset/Model:** OLMoE-1B-7B
- **Sequence Length:** 512 (Standardized). Note: Initial iterations evaluated 256-token sequences (Legacy).
- **Design:** Evaluated 7 univariate pre-merge features across structural layers (`first`, `middle`, `last`).
- **Ground Truth:** Oracle KL Divergence upon averaging candidate pairs.

### Equations
- **Definition (Capability Response):** $C_i \in \mathbb{R}^{10}$
- **Definition (Token Environment):** $\tau_i \in \mathbb{R}_+^{10}$
- **Derivation (Weight Distance):** $D_{ij} = ||W_i - W_j||_2$
- **Prediction:** High similarity (low $D_{ij}$) predicts low functional degradation (Oracle KL $pprox 0$).

### Plots
- ![Weight Distance (First Layer)](./results/exp1/256_segmented/first/scatter_first_Weight_Distance.png)
- ![Weight Cosine (Middle Layer)](./results/exp1/128_segmented/middle/scatter_middle_Weight_Cosine.png)
- ![Activation Similarity (Last Layer)](./results/exp1/64_segmented/last/scatter_last_Activation_Similarity.png)

### Output
- **Spearman Rank ($
ho$):** $|
ho| < 0.2$ for almost all metrics across all layers.
- **Visual Distribution:** Isotropic point clouds rather than compact monotonic bands.

### Conclusion
**Hypothesis Rejected.** No single metric accurately predicts functional degradation. Weight distance sets a loose outer boundary, but capability preservation is a latent emergent property dependent on structural weights, utilization frequency, and contextual gating.

---

## Experiment 1.5: Multivariate Capability Modeling

### Hypothesis
Combining multiple weak univariate pre-merge features via linear (LASSO) and non-linear (XGBoost) models can successfully predict Oracle KL drift and safeguard expert merging.

### Experiment
- **Dataset/Model:** OLMoE-1B-7B
- **Sequence Length:** 512 (Standardized). (Legacy evaluations at 256).
- **Design:** Disjoint expert partition (Train: 0-31, Test: 32-63) to prevent identity leakage. Purged all oracle-grade features. Evaluated 12 model configurations.

### Equations
- **Definition (Linearization Gap):** $\Delta = 
ho_{	ext{tree}} - 
ho_{	ext{linear}}$
- **Prediction:** Multivariate models will yield high test $R^2$ scores, and non-linear trees will outperform linear hyperplanes.

### Plots
- ![Linearization Gap Across Models](./results/exp1_5/figures/06_linearization_gap.png)
- ![XGBoost Gain Importance](./results/exp1_5/figures/03_xgboost_importance.png)
- ![Predicted vs Actual Scatter](./results/exp1_5/figures/04_predicted_vs_oracle.png)

### Output
- **Best Non-Linear Model (XGBoost):** $
ho = 0.593$, Test $R^2 = -0.507$ (Catastrophic Out-of-Distribution Calibration).
- **Best Linear Model (LASSO):** $
ho = 0.484$, Test $R^2 = 0.037$.
- **Linearization Gap:** $\Delta = +0.109$.

### Conclusion
**Hypothesis Rejected.** Existing pre-merge features are fundamentally insufficient. The massive Linearization Gap reveals that non-additive, depth-dependent feature interactions govern routing, but current features fail to predict high-drift outliers (tail-blindness).

---

## Experiment 2: Capability-Aware Descriptor Engineering

### Hypothesis
Engineered pre-merge descriptors that capture asymmetric dominance, functional co-activation (NPMI), and usage divergence can resolve the predictive deficit and close the linearization gap across network depth.

### Experiment
- **Dataset/Model:** OLMoE-1B-7B
- **Design:** Engineered four new variables: Usage Asymmetry, Routing JSD Proxy, Routing NPMI Proxy, and Specialization Diff. Evaluated via stratified within-layer Phase 6 analysis.

### Equations
- **Definition (Usage Asymmetry):** $\Delta_{	ext{usage}} = |ar{u}_i - ar{u}_j|$
- **Definition (NPMI Proxy):** $	ext{NPMI} = 	ext{clip}\left( \frac{\log(P(i, j) / (P(i)P(j)))}{-\log P(i, j)}, -1, +1 
ight)$
- **Prediction:** Engineered functional co-activation will outrank classical weight geometry in splitting gain.

### Plots
- ![XGBoost Gain Importance](./results/exp2/plots/shap/xgboost_importance.png)
- ![SHAP Beeswarm Summary](./results/exp2/plots/shap/shap_summary.png)
- ![Linearization Gap Comparison](./results/exp2/plots/regression/gap_comparison.png)

### Output
- **NPMI Dominance:** `Routing_NPMI_Proxy` controlled 15.98% of total XGBoost split gain.
- **Stratified Gap:** 
  - `first` layer: $\Delta = +0.3399$
  - `middle` layer: $\Delta = +0.0185$
  - `last` layer: $\Delta = +0.0195$ ($
ho > 0.83$)

### Conclusion
**Hypothesis Supported.** Engineered features dramatically improve prediction. Furthermore, the Linearization Gap is not global; it is highly localized. Initial layers exhibit severe non-linear gating thresholds, while deeper layers converge to high-fidelity linear predictability.

---

## Experiment 3A: Global Functional Communities

### Hypothesis
Experts organize globally into discrete, highly intra-connected functional communities rather than operating as uniform, independent sub-networks.

### Experiment
- **Design:** Scaled pairwise NPMI capability proxies to an $N \times N$ adjacency matrix. Applied Louvain Modularity clustering across `first`, `middle`, and `last` layers.

### Equations
- **Definition (Modularity):** $Q = \frac{1}{2m} \sum_{i,j} \left[ A_{i,j} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$
- **Prediction:** Network Modularity $Q$ will vary by layer depth, indicating non-uniform community structure.

### Plots
- (Plots available in the dedicated Exp 3A report directory).

### Output
- **Layer First:** $Q = 0.084$ (Monolithic)
- **Layer Middle:** $Q = 0.203$ (Maximum modularity; 4-5 sub-spaces)
- **Layer Last:** $Q = 0.126$ (Collapsing modularity)

### Conclusion
**Observational Finding.** Expert relationships exhibit non-uniform community structure, with the strongest modular organization appearing in the middle layer. Note: Statistical significance relative to a null distribution was not established in the current analysis; this remains an exploratory topological observation.

---

## Experiment 3B: Capability Geometry Validation

### Hypothesis
The true functional behavior of MoE experts possesses a continuous, low-dimensional structured geometry that generalizes to unseen experts.

### Experiment
- **Design:** 50-fold out-of-sample Non-Metric Multidimensional Scaling (SMACOF) on ground-truth Oracle KL divergence.
- **Null Models:** Evaluated against 30 Random-Euclidean (Null B) and Pairwise-Shuffled (Null A) models.

### Equations
- **Definition (Objective):** Stress $\sigma = \sqrt{\sum (D_{ij} - ||Z_i - Z_j||)^2}$
- **Prediction:** Out-of-sample Euclidean distance in the $Z$-space will correlate highly with Oracle KL for held-out experts.

### Plots
- (Plots available in the dedicated Exp 3B report directory).

### Output
- **Middle Layer ($q=4$):** Oracle out-of-sample $\rho = +0.723$, outperforming Null A ($\rho = 0.015$) and Null B ($\rho = 0.298$).
- **Dimensionality:** `first` requires $q=3-4$, `middle` requires $q=4$, `last` requires $q=8-9$.

### Conclusion
**Predictive Finding.** Expert capabilities exhibit a low-dimensional, continuously structured functional geometry. Note: The evidence supports predictive geometric organization, but does not establish a globally smooth manifold in the strict mathematical sense.

---

## Experiment 3C: Capability Geometry Evolution

### Hypothesis
The functional geometric relationships between experts evolve continuously over training time, exhibiting layer-dependent differentiation trajectories.

### Experiment
- **Checkpoints:** 10%, 40%, 70%, 100%.
- **Design:** Sparse 3C pair manifest evaluated via weighted SMACOF and Procrustes alignment to extract continuous velocity fields.

### Equations
- **Definition (Procrustes Alignment):** $Z_{aligned} = Z \cdot Q + t$ minimizing squared differences.
- **Prediction:** Expert geometries will show structured consistency over time, alongside expanding pairwise distances.

### Plots
- (Plots available in the dedicated Exp 3C report directory).

### Output
- **First/Last Layers:** Continuous monotonic separation (increasing Oracle KL).
- **Middle Layer:** U-shaped trajectory (distances drop from 10% to 70%, then expand).

### Conclusion
**Observational Finding.** Functional relationships remain structured across checkpoints as experts undergo continuous differentiation. Layer-dependent trajectories are observed: the first and last layers expand monotonically, while the middle layer shows an initial redundancy bottleneck (U-shaped trajectory).

---

## Experiment 4: The Functional Merge Landscape

### Hypothesis
Geometric distance extracted from the capability space provides complementary predictive information regarding functional merge damage, and combining local features with geometry (CARE) alters predictive behavior relative to local or geometric models alone.

### Experiment
- **Design:** 5-partition $\times$ 3-fold cross-validation. Compared XGBoost on local features (Model A), pure Geometry distance (Model B), and CARE (Model C: Local + Geometry).

### Equations
- **Prediction:** $\rho_C$ and $\rho_B$ will diverge from $\rho_A$, confirming geometry provides non-redundant predictive gain.

### Plots
- ![Spearman Correlation by Model](./results/exp4/plots/01_spearman_by_model.png)
- ![Delta Rho Distribution](./results/exp4/plots/03_delta_rho_distribution.png)
- ![Precision at K](./results/exp4/plots/05_precision_at_k.png)

### Output
- **Model A (Local):** $\rho = 0.4797$
- **Model B (Geometry):** $\rho = 0.7504$
- **Model C (Local+Geometry):** $\rho = 0.8146$
- **Precision@10:** A=0.17, B=0.37, C=0.23
- **Precision@25:** A=0.36, B=0.50, C=0.41
- **Precision@50:** A=0.52, B=0.63, C=0.69

### Conclusion
**Hypothesis Supported.** Geometric distance provides substantial predictive information about functional merge damage. However, in the highly selective top-$K$ regime ($K=10$, $K=25$), the pure geometry model outperforms the combined CARE descriptor, whereas the combined model only becomes strongest at larger $K$ ($K=50$). This indicates that the utility of local descriptors is budget-dependent rather than uniformly additive.

---

## Experiment 5: Functional Merge Execution

### Hypothesis
Merging redundant experts guided by the CARE geometric metric will result in significantly lower capability degradation compared to standard L2 weight distance or naive usage heuristics.

### Experiment
- **Design:** Executed actual parameter consolidation on the model. Evaluated degradation across layers using different heuristic ranking strategies.

### Equations
- **Definition (Merge):** $W_{merged} = \frac{W_i + W_j}{2}$
- **Prediction:** CARE-guided merges will maintain lower overall network perplexity and benchmark degradation than naive merges.

### Plots
- (Plots available in the dedicated Exp 5 report directory).

### Output
- **Performance:** CARE geometric selection resulted in substantially lower functional degradation than L2 distance baselines.
- **Layer Limits:** Deep layers exhibited high tolerance for parameter consolidation, while initial structural layers degraded rapidly.

### Conclusion
**Hypothesis Supported.** The geometric capability metric translates successfully from theoretical proxy to actionable compression algorithm, providing superior protection against catastrophic merge damage.

---

## Experiment 6B & 6C: Observational Functional Dynamics

### Hypothesis
An expert's tangential functional displacement $\Delta C_\perp$ across training is directionally guided by the orthogonal component of its interaction with the token environment ($I = C \odot \tau$).

### Experiment
- **Design:** Mapped functional vectors $C_i$ (capability probe response) and environments $\tau_i$. Decomposed movement into radial ($\Delta C_\parallel$) and tangential ($\Delta C_\perp$).
- **Correction:** Excluded pre-release probing bug data.

### Equations
- **Definition (Displacement):** $\Delta C_i = C_i(t+1) - C_i(t)$
- **Decomposition:** $\Delta C_i = \Delta C_{i, \parallel} + \Delta C_{i, \perp}$
- **Interaction (Hypothesis):** $I_i = C_i \odot \tau_i$
- **Prediction:** $R^2(I_\perp \to \Delta C_\perp)$ is statistically significant.

### Plots
- (Plots available in the Exp 6C report directory).

### Output
- **Global Movement:** Dominated by radial magnitude contraction.
- **Late-Stage Tangential:** $R^2(I_\perp \to \Delta C_\perp) = 0.2542$ (Significant, $Z = -3.81$).
- **Divergence:** Task-Overlap correlates with positive $\Delta D$ ($
ho pprox 0.50$).

### Conclusion
**Hypothesis Supported (with limitations).** While radial contraction dominates global variance, the specific expert-environment interaction $I$ accounts for a mathematically verifiable portion ($\sim 25\%$) of the tangential task-specific steering during late training. Experts processing similar tasks actively diverge.

---

## Experiment 6D: Controlled Interventional Dynamics

### Hypothesis
Actively perturbing an expert's training environment $\tau$ along controlled structural angles will induce predictable, magnitude- and direction-dependent functional drift.

### Experiment
- **Design:** 900-condition GPU sweep. Intervened during training by scaling target expert loss ($loss' = \alpha \times loss$) while keeping token subsets constant.
- **Parameters:** $\alpha \in [0.01, 5.0]$, controlled orthogonal target angles $\theta$.

### Equations
- **Definition (Intervention):** $loss' = \alpha \times loss$
- **Prediction:** The angular functional drift $\Delta\theta$ will scale with intervention strength $\alpha$, and the model will resist orthogonal shifts more than aligned shifts.

### Plots
- ![Linearity in Low Alpha](./results/exp6d_rerun/exp6d/plots/10_low_alpha_linearity.png)
- ![Delta Theta Curves](./results/exp6d_rerun/exp6d/plots/08_delta_theta_curves_by_angle.png)
- ![Directional Response by Quantile](./results/exp6d_rerun/exp6d/plots/09_directional_response_by_quantile.png)

### Output
- **Low-Alpha:** $\Delta\theta$ scales approximately linearly in the tested low-$\alpha$ regime ($\alpha \le 1.0$).
- **Directional Resistance:** Intervening at maximum orthogonal angles causes ~2.3x more functional drift than aligned interventions at the exact same $\alpha$ magnitude.
- **State-Dependence:** High $||C||$ experts strongly resist structural drift.

### Conclusion
**Interventional Evidence.** Controlled interventions provide evidence that functional responses are direction-dependent, magnitude-dependent, and state-dependent on initial capability magnitude. Note: These observations support a local geometric model but do not establish a globally valid continuous manifold.

---

## Overarching Final Conclusion

The empirical evidence from Experiments 1 through 6 demonstrates that MoE expert capabilities exhibit a structured, functional geometry that is layer-dependent and evolves predictably over time. 

- **Observational:** We observe clear topological patterns, including low-dimensional geometric structure (Exp 3B) and varying modularity (Exp 3A) that peaks in the middle layers.
- **Predictive:** Geometric features extracted from this space are highly predictive of functional merge damage, providing a robust baseline for budget-constrained compression (Exp 4). 
- **Interventional:** Controlled structural interventions reveal that the network's functional responses are highly direction- and magnitude-dependent, confirming the presence of local geometric constraints (Exp 6D).

Together, these findings advance the understanding of MoE internal representation from unstructured sets of parameters to organized functional geometries, laying the foundation for "Interpretability as a Science."

> [!WARNING]
> **Limitation:** All experiments were conducted on a single open-source model architecture (`OLMoE-1B-7B-0924`). While the internal statistics are highly robust across random seeds and cross-validation folds, we do not claim that these specific geometric properties generalize to all MoE architectures. Future work is required to determine whether these topological phenomena are broad features of MoE training or specific to this model's routing mechanism.
