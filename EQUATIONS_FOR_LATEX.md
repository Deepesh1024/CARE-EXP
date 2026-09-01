# EQUATIONS FOR LATEX — Authoritative Equation Sheet

> **Purpose:** This document is the single source of truth for every equation
> in the CARE-MoE / Interpretability-as-a-Science paper.
> It is intended for the two interns converting the paper into LaTeX.
> Every equation has been verified against the implementation in the repository.
> Do NOT copy equations from other reports; use this file.

---

## 1. CAPABILITY DEFINITION

**Type:** Definition

$$
C_i = \bigl(c_{i,1},\; c_{i,2},\; \dots,\; c_{i,d}\bigr) \in \mathbb{R}^{d}, \qquad d = 10
$$

**Plain English:** The capability vector $C_i$ is the empirical functional response of expert $i$. Each component $c_{i,k}$ is the mean $L_2$ norm of the expert's output activation when driven by tokens from semantic category $k$.

**Component definition:**

$$
c_{i,k} = \frac{1}{|S_k|} \sum_{x \in S_k} \frac{1}{T_x} \left\| E_i(h^{(l)}(x)) \right\|_2
$$

where $S_k$ is the token pool for capability axis $k$, $T_x$ is the valid sequence length of token $x$, $E_i$ is the expert MLP forward pass, and $h^{(l)}(x)$ is the hidden state entering layer $l$.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $C_i$ | Capability vector of expert $i$ |
| $c_{i,k}$ | Capability of expert $i$ on axis $k$ |
| $d$ | Capability-space dimensionality ($d = 10$) |
| $S_k$ | Token pool for capability axis $k$ |
| $T_x$ | Valid sequence length of token $x$ |
| $E_i$ | Expert $i$ MLP forward function |
| $h^{(l)}(x)$ | Hidden state at layer $l$ for input $x$ |

**Source:** `experiments/experiment6d/capability_probe.py` lines 16–56; `experiments/experiment6c/config.py` lines 65–116.

**Status:** [VERIFIED]

---

## 2. FUNCTIONAL DISTANCE

**Type:** Definition

$$
D^{C}_{ij} = \| C_i - C_j \|_2
$$

**Plain English:** The functional distance between experts $i$ and $j$ is the Euclidean distance between their capability vectors. This measures how differently two experts respond across the 10 semantic task axes.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $D^{C}_{ij}$ | Functional distance between experts $i$ and $j$ |

**Source:** `experiments/experiment3b/phase1_distance_matrix.py` (Oracle distance matrix built from $D^C_{ij}$ values).

**Status:** [VERIFIED]

---

## 3. WEIGHT-SPACE COMPARISON

**Type:** Definition

### 3a. Weight Distance

$$
D^{W}_{ij} = \| W_i - W_j \|_2
$$

**Plain English:** The Euclidean distance between the flattened parameter vectors of experts $i$ and $j$.

### 3b. Weight Cosine Similarity

$$
\text{WeightCos}(i, j) = \frac{W_i \cdot W_j}{\| W_i \| \; \| W_j \|}
$$

**Plain English:** The cosine similarity between the flattened parameter vectors.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $W_i$ | Flattened parameter vector of expert $i$ (concatenation of `gate_up_proj` and `down_proj`) |
| $D^{W}_{ij}$ | Parameter-space Euclidean distance |

**Implementation detail:** For the fused OLMoE architecture, $W_i$ is the concatenation of the `gate_up_proj` and `down_proj` weights for expert $i$.

**Source:** `experiments/experiment1/CARE_MoE_V3_E1.py` lines 500–508 (weight extraction), line 592 (Weight_Distance), line 593 (Weight_Cosine).

**Status:** [VERIFIED]

---

## 4. ACTIVATION SIMILARITY

**Type:** Definition

$$
\text{ActSim}(i, j) = \frac{1}{T} \sum_{t=1}^{T} \frac{a_i^{(t)} \cdot a_j^{(t)}}{\| a_i^{(t)} \| \; \| a_j^{(t)} \|}
$$

**Plain English:** The activation similarity is the **mean token-level cosine similarity** between the input activations received by experts $i$ and $j$, averaged across all $T$ calibration tokens. Each $a_i^{(t)}$ is the activation vector captured at the input to expert $i$ for token $t$.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $a_i^{(t)}$ | Activation (input hidden state) received by expert $i$ at token $t$ |
| $T$ | Number of calibration tokens |

**Source:** `experiments/experiment1/CARE_MoE_V3_E1.py` line 594:
```python
F.cosine_similarity(expert_activations[i], expert_activations[j], dim=-1).mean().item()
```

**Status:** [VERIFIED]

---

## 5. FUNCTIONAL DEGRADATION / ORACLE KL

**Type:** Definition

$$
\mathcal{L}_{\text{oracle}}(i, j) = \frac{1}{T} \sum_{t=1}^{T} D_{\text{KL}}\!\left( P_{\text{orig}}(\cdot | x_t) \;\|\; P_{\text{merged}}^{(i,j)}(\cdot | x_t) \right)
$$

where:

$$
D_{\text{KL}}(P \| Q) = \sum_{v \in \mathcal{V}} P(v) \log \frac{P(v)}{Q(v)}
$$

**Plain English:** The Oracle KL is the mean per-token KL divergence between the original model's next-token prediction distribution and the prediction distribution after merging experts $i$ and $j$ via uniform parameter averaging. Only non-padding (shift-masked) tokens are included.

**Implementation detail:** The merge operator replaces both experts $i$ and $j$ with $E_{i+j}$ where $W_{i+j} = \frac{1}{2}(W_i + W_j)$. Logits are shifted by one position (autoregressive convention). KL is computed as:
```python
logp_orig = F.log_softmax(shift_logits_orig.float(), dim=-1)
logp_merged = F.log_softmax(shift_logits_merged.float(), dim=-1)
kl_tok = (logp_orig.exp() * (logp_orig - logp_merged)).sum(dim=-1)
Oracle_KL = kl_sum / total_tokens
```

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $\mathcal{L}_{\text{oracle}}(i,j)$ | Mean per-token Oracle KL divergence for the merge of experts $i, j$ |
| $P_{\text{orig}}$ | Next-token distribution of the unmodified model |
| $P_{\text{merged}}^{(i,j)}$ | Next-token distribution after merging experts $i$ and $j$ |
| $T$ | Number of valid (non-padding) tokens |
| $\mathcal{V}$ | Vocabulary |

**Source:** `experiments/experiment1/CARE_MoE_V3_E1.py` lines 311–315, 351.

**Status:** [VERIFIED]

---

## 6. MULTIVARIATE CAPABILITY PREDICTION

**Type:** Definition

$$
\hat{D}^{\text{KL}}_{ij} = f_\theta(X_{ij})
$$

where $X_{ij} \in \mathbb{R}^{p}$ is the engineered feature vector for pair $(i, j)$, and $f_\theta$ is the learned surrogate predictor.

**Plain English:** The predicted merge damage for a pair $(i, j)$ is the output of a surrogate model applied to the pre-merge feature descriptor.

**Feature variants used:**
- **Model A:** $p = 11$ local pre-merge features (XGBoost regressor)
- **Model B:** $p = 1$ — the single geometric distance $D^{C}_{ij}$ in MDS space (not a learned predictor)
- **Model C (CARE):** $p = 12$ — 11 local features + 1 geometric distance (XGBoost regressor)

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $X_{ij}$ | Engineered feature vector for expert pair $(i, j)$ |
| $f_\theta$ | Learned surrogate predictor (XGBoost or LASSO) |
| $\hat{D}^{\text{KL}}_{ij}$ | Predicted Oracle KL for pair $(i, j)$ |
| $p$ | Number of features in the descriptor |

**Source:** `experiments/experiment4/config.py`; `results/exp4/final_report.md`.

**Status:** [VERIFIED]

---

## 7. CAPABILITY GEOMETRY

**Type:** Definition (embedding) + Derived (stress objective)

### 7a. MDS Embedding

$$
C_i \mapsto z_i \in \mathbb{R}^{q}
$$

such that:

$$
\| z_i - z_j \|_2 \approx D^{C}_{ij} \quad \forall\; i, j
$$

**Plain English:** Each expert's capability vector is mapped to a low-dimensional coordinate $z_i$ via Multidimensional Scaling, preserving pairwise functional distances.

### 7b. SMACOF Stress Objective

The implementation uses scikit-learn's metric SMACOF algorithm, which minimizes the raw stress:

$$
\sigma_{\text{raw}}(Z) = \sum_{i < j} \left( \| z_i - z_j \|_2 - D^{C}_{ij} \right)^2
$$

**Implementation parameters:** `n_init=4`, `max_iter=3000`, `eps=1e-4`, `metric=True`.

### 7c. Kruskal's Normalized Stress-1

The validation statistic reported in Exp 3B is Kruskal's stress-1:

$$
\sigma_1(Z) = \sqrt{ \frac{\sum_{i < j} \left( \| z_i - z_j \| - D^{C}_{ij} \right)^2}{\sum_{i < j} \left( D^{C}_{ij} \right)^2} }
$$

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $z_i$ | Low-dimensional MDS coordinate for expert $i$ |
| $q$ | Embedding dimensionality ($q = 4$ selected in Exp 3B) |
| $\sigma_{\text{raw}}$ | Raw SMACOF stress |
| $\sigma_1$ | Kruskal's normalized stress-1 |

**Source:** `experiments/experiment3b/phase2_mds_nulls.py` lines 58–107 (SMACOF), lines 110–129 (normalized stress); `experiments/experiment3b/config.py` lines 51–55.

**Status:** [VERIFIED]

---

## 8. NULL / GEOMETRY VALIDATION

**Type:** Empirical statistic

The validation in Experiment 3B compares the normalized stress of the real Oracle distance matrix against two null distributions:

- **Null A (Pairwise Shuffle):** Permute all upper-triangle distances, preserving the marginal distance distribution but destroying expert-identity structure.
- **Null B (Gaussian i.i.d.):** Replace each upper-triangle entry with $\mathcal{N}(\mu_D, \sigma_D^2)$ drawn from the empirical mean and variance of the real distances.

The reported statistic is:

$$
\Delta\sigma_1 = \sigma_1^{\text{null}} - \sigma_1^{\text{real}}
$$

**Plain English:** If $\Delta\sigma_1 > 0$ for all null realizations, the real data embeds substantially better than chance, supporting the existence of low-dimensional geometric structure.

**Source:** `experiments/experiment3b/phase2_mds_nulls.py` lines 132–175 (null generation); `results/exp3b/final_report.md`.

**Status:** [VERIFIED]

---

## 9. CAPABILITY EVOLUTION

**Type:** Definition

$$
C_i^{(l, t)} \in \mathbb{R}^{d}
$$

**Plain English:** The capability vector of expert $i$ at layer $l$ and training checkpoint $t$.

$$
D_{ij}^{(l, t)} = \| C_i^{(l, t)} - C_j^{(l, t)} \|_2
$$

**Plain English:** The functional distance between experts $i$ and $j$ at layer $l$ and checkpoint $t$.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $l$ | Layer index ($l \in \{0, 1, \dots, 15\}$) |
| $t$ | Training checkpoint index |
| $C_i^{(l,t)}$ | Capability of expert $i$ at layer $l$, checkpoint $t$ |

> [!NOTE]
> The trajectory $\gamma_i^{(l)}(t)$ is used informally in the Exp 3C prose to describe the path of an expert through capability space over training time, but it is NOT implemented as an explicit continuous-time parametric curve. The analysis uses discrete checkpoint comparisons with Procrustes alignment via `scipy.spatial.procrustes`.

**Source:** `experiments/experiment3c/phase4_analysis.py` line 182; `experiments/experiment6c/config.py` lines 46–51 (checkpoint definitions).

**Status:** [VERIFIED] — trajectory $\gamma$ is informal notation, not a computed object.

---

## 10. FUNCTIONAL DISPLACEMENT

**Type:** Definition (displacement) + Derived (decomposition)

$$
\Delta C_i = C_i(t+1) - C_i(t)
$$

**Plain English:** The change in capability between consecutive checkpoints.

### 10a. Parallel Component

$$
\Delta C_{\parallel} = \left( \Delta C \cdot \hat{C} \right) \hat{C}
$$

where $\hat{C} = C_i / \| C_i \|$ is the unit capability direction.

### 10b. Perpendicular Component

$$
\Delta C_{\perp} = \Delta C - \Delta C_{\parallel}
$$

**Plain English:** The parallel component captures capability magnitude change (growth/shrinkage); the perpendicular component captures functional reorientation.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $\Delta C_i$ | Capability displacement vector |
| $\hat{C}$ | Unit capability direction: $C / \|C\|$ |
| $\Delta C_{\parallel}$ | Projection of displacement along $\hat{C}$ |
| $\Delta C_{\perp}$ | Component of displacement orthogonal to $\hat{C}$ |

**Source:** `experiments/experiment6d/intervention.py` lines 266–267:
```python
dc_par = np.dot(delta_c, c_hat) * c_hat
dc_perp = delta_c - dc_par
```

**Status:** [VERIFIED]

---

## 11. ENVIRONMENT / INTERVENTION VECTOR

**Type:** Definition

$$
\tau \in \mathbb{R}^{d}, \qquad d = 10
$$

$$
\tau = \| \tau \| \; \hat{\tau}
$$

**Plain English:** The intervention (environment) vector $\tau$ encodes the semantic composition of the training signal applied to an expert. Its magnitude controls total training pressure; its direction encodes which capability axes are emphasised.

### 11a. Orthogonal Component

$$
\tau_{\perp} = \tau - \frac{\tau^\top C}{\| C \|^2} C
$$

$$
\| \tau_{\perp} \| = \| \tau \| \sin\theta
$$

where $\theta$ is the angle between $\tau$ and $C$:

$$
\theta = \arccos\left( \frac{\tau^\top C}{\| \tau \| \; \| C \|} \right)
$$

**Plain English:** $\theta$ is the angle between the intervention direction and the expert's current capability direction. When $\theta = 0$, the intervention is fully aligned; when $\theta = 90^\circ$, it is fully orthogonal.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $\tau$ | Intervention/environment vector |
| $\hat{\tau}$ | Unit direction of intervention |
| $\theta$ | Angle between intervention $\tau$ and capability $C$ |
| $\tau_{\perp}$ | Component of $\tau$ orthogonal to $C$ |

**Source:** `experiments/experiment6d/intervention.py` line 253 (alpha-scaled loss drives tau); `results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md`.

**Status:** [VERIFIED]

---

## 12. SUSCEPTIBILITY HYPOTHESIS

**Type:** Experimental Hypothesis (NOT a proven universal law)

**Definition:**

$$
S(C, \tau) = \frac{\| \tau_{\perp} \|}{\| C \|}
$$

**Predicted relationship:**

$$
\Delta\theta \;\propto\; \frac{\| \tau_{\perp} \|}{\| C \|}
$$

**Plain English:** The susceptibility hypothesis predicts that the angular displacement $\Delta\theta$ of an expert's capability vector scales with the ratio of the orthogonal intervention component to the expert's current capability magnitude. Experts with large $\|C\|$ are predicted to be more resistant.

> [!WARNING]
> This is an **empirical hypothesis** supported by the Exp 6D data in the tested regime ($\alpha \le 1.0$). It is NOT a proven universal law and should NOT be stated as one.

**Source:** `results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md`.

**Status:** [VERIFIED] (as hypothesis)

---

## 13. LOSS-SCALED INTERVENTION (EXPERIMENT 6D)

**Type:** Derived mathematical identity + Implementation-specific

### 13a. Loss Scaling

$$
\mathcal{L}_\alpha = \alpha \cdot \mathcal{L}
$$

**Plain English:** The loss is scaled by a constant factor $\alpha$ before backpropagation.

### 13b. Gradient Scaling

$$
\nabla_\phi \mathcal{L}_\alpha = \alpha \; \nabla_\phi \mathcal{L}
$$

**Plain English:** By the chain rule, scaling the loss by $\alpha$ scales all gradients by $\alpha$.

### 13c. SGD Parameter Update

The implementation uses vanilla SGD (no momentum):

$$
\phi \leftarrow \phi - \eta \, \alpha \, \nabla_\phi \mathcal{L}
$$

where $\eta$ is the learning rate.

**Implementation detail:** The actual code computes the loss, multiplies by $\alpha$, and calls `.backward()`:
```python
optimizer = torch.optim.SGD(..., lr=5e-4)
loss = (outputs.loss / (BATCH_SIZE // MICRO_BATCH_SIZE)) * row["alpha"]
loss.backward()
optimizer.step()
```
The loss is also divided by the gradient accumulation factor `BATCH_SIZE // MICRO_BATCH_SIZE` before scaling by $\alpha$.

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $\phi$ | Model parameters (only the targeted expert's weights are trainable) |
| $\alpha$ | Intervention magnitude scalar |
| $\eta$ | Learning rate ($5 \times 10^{-4}$) |
| $\mathcal{L}$ | Cross-entropy loss on the intervention tokens |

**Source:** `experiments/experiment6d/intervention.py` lines 200, 249–256; `experiments/experiment6d/config.py` line 31 (LR), line 33 (BATCH_SIZE).

**Status:** [VERIFIED]

---

## 14. LOCAL FUNCTIONAL RESPONSE

**Type:** Derived mathematical identity (local linear approximation)

$$
\Delta C \approx J_C(\phi) \, \Delta\phi
$$

where:

$$
J_C(\phi) = \frac{\partial C}{\partial \phi}
$$

**Plain English:** To first order, the change in capability is the Jacobian of the capability function with respect to model parameters, multiplied by the parameter update.

Substituting the SGD update from §13c:

$$
\Delta C \approx -\eta \, \alpha \; J_C(\phi) \, \nabla_\phi \mathcal{L}
$$

> [!IMPORTANT]
> The Jacobian $J_C(\phi)$ is **never explicitly computed** in the experiments. This equation is the theoretical justification for predicting that $\Delta C$ should scale linearly with $\alpha$ in the small-$\alpha$ regime. The empirical evidence (Exp 6D low-$\alpha$ linearity) supports this relationship but does not prove it holds globally.

**Source:** Theoretical derivation; empirical support from `results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md`.

**Status:** [VERIFIED] (as local approximation, not global identity)

---

## 15. CAPABILITY ANGULAR DISPLACEMENT

**Type:** Definition

$$
\delta\theta = \arccos\left( \frac{C_{\text{before}}^\top \, C_{\text{after}}}{\| C_{\text{before}} \| \; \| C_{\text{after}} \|} \right)
$$

**Plain English:** The angular displacement $\delta\theta$ is the angle between the capability vector before and after intervention.

**Implementation:**
```python
def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))
```

> [!WARNING]
> **[AUDIT REQUIRED]** The implementation returns the angle in **degrees** (`np.degrees`), but the LaTeX paper and some reports reference $\delta\theta$ in radians. The raw data in `EXP6D_RAW_RESULTS.parquet` column `delta_theta` is stored in **degrees**. The interns must decide whether to convert to radians for the paper or clearly label the unit.

**Source:** `experiments/experiment6d/intervention.py` lines 51–57, line 297.

**Status:** [AUDIT REQUIRED] — degree vs radian unit convention must be explicitly resolved.

---

## 16. APPENDIX / OPTIONAL EQUATIONS

### 16a. Modularity (Experiment 3A)

**Type:** Definition (standard graph theory)

$$
Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

**Plain English:** The Newman-Girvan modularity $Q$ measures the fraction of edges within communities minus the expected fraction under a configuration model null.

**Implementation:** The code uses the `community_louvain` package from `community` (python-louvain), which implements the standard Louvain algorithm on an **unweighted binary graph** constructed by thresholding the capability distance matrix.

```python
community_louvain.modularity(partition, G_binary)
```

**Symbol definitions:**
| Symbol | Meaning |
|---|---|
| $A_{ij}$ | Adjacency matrix entry (binary) |
| $k_i$ | Degree of node $i$ |
| $m$ | Total number of edges |
| $c_i$ | Community assignment of node $i$ |
| $\delta(c_i, c_j)$ | Kronecker delta: 1 if same community, 0 otherwise |

**Source:** `experiments/experiment3a/phase3_community_detection.py` line 61; `experiments/experiment3a/exp3a_null_a.py` line 61.

**Status:** [VERIFIED]

---

### 16b. Expert Merge (Experiment 1 / 5)

**Type:** Definition

$$
W_{i+j} = \frac{1}{2} \left( W_i + W_j \right)
$$

**Plain English:** The merged expert $E_{i+j}$ is created by element-wise averaging of all parameter tensors of experts $i$ and $j$. Both slot $i$ and slot $j$ in the MoE layer receive the same averaged weights.

**Implementation (fused architecture):**
```python
merged_gate = 0.5 * (gate_i + gate_j)
merged_down = 0.5 * (down_i + down_j)
```

**Source:** `experiments/experiment1/CARE_MoE_V3_E1.py` lines 254–255.

**Status:** [VERIFIED]

---

## NOTATION TABLE

| Symbol | Meaning | Dimensions | Source |
|---|---|---|---|
| $C_i$ | Capability vector of expert $i$ | $\mathbb{R}^{10}$ | §1 |
| $c_{i,k}$ | Capability of expert $i$ on axis $k$ | scalar | §1 |
| $d$ | Capability-space dimensionality | $d = 10$ | §1 |
| $D^{C}_{ij}$ | Functional distance between experts | scalar | §2 |
| $D^{W}_{ij}$ | Weight-space Euclidean distance | scalar | §3 |
| $W_i$ | Flattened parameter vector of expert $i$ | $\mathbb{R}^{p_W}$ | §3 |
| $a_i^{(t)}$ | Input activation to expert $i$ at token $t$ | $\mathbb{R}^{h}$ | §4 |
| $\mathcal{L}_{\text{oracle}}(i,j)$ | Oracle KL divergence for merge $(i,j)$ | scalar | §5 |
| $P_{\text{orig}}$ | Original model next-token distribution | $\Delta^{|\mathcal{V}|}$ | §5 |
| $P_{\text{merged}}^{(i,j)}$ | Merged model next-token distribution | $\Delta^{|\mathcal{V}|}$ | §5 |
| $T$ | Number of valid tokens | integer | §5 |
| $\mathcal{V}$ | Vocabulary | set | §5 |
| $X_{ij}$ | Engineered feature vector for pair $(i,j)$ | $\mathbb{R}^{p}$ | §6 |
| $f_\theta$ | Learned surrogate predictor | function | §6 |
| $\hat{D}^{\text{KL}}_{ij}$ | Predicted Oracle KL | scalar | §6 |
| $z_i$ | MDS embedding coordinate | $\mathbb{R}^{q}$ | §7 |
| $q$ | Embedding dimensionality | $q = 4$ | §7 |
| $\sigma_{\text{raw}}$ | Raw SMACOF stress | scalar | §7 |
| $\sigma_1$ | Kruskal's normalized stress-1 | scalar | §7 |
| $l$ | Layer index | $l \in \{0, \dots, 15\}$ | §9 |
| $t$ | Training checkpoint index | integer | §9 |
| $C_i^{(l,t)}$ | Capability at layer $l$, checkpoint $t$ | $\mathbb{R}^{10}$ | §9 |
| $\Delta C_i$ | Capability displacement | $\mathbb{R}^{10}$ | §10 |
| $\hat{C}$ | Unit capability direction | $\mathbb{R}^{10}$ | §10 |
| $\Delta C_{\parallel}$ | Parallel displacement component | $\mathbb{R}^{10}$ | §10 |
| $\Delta C_{\perp}$ | Perpendicular displacement component | $\mathbb{R}^{10}$ | §10 |
| $\tau$ | Intervention/environment vector | $\mathbb{R}^{10}$ | §11 |
| $\hat{\tau}$ | Unit intervention direction | $\mathbb{R}^{10}$ | §11 |
| $\theta$ | Angle between $\tau$ and $C$ | radians/degrees | §11 |
| $\tau_{\perp}$ | Orthogonal component of intervention | $\mathbb{R}^{10}$ | §11 |
| $S(C, \tau)$ | Susceptibility | scalar | §12 |
| $\alpha$ | Intervention magnitude scalar | scalar | §13 |
| $\phi$ | Model parameters (targeted expert) | $\mathbb{R}^{p_\phi}$ | §13 |
| $\eta$ | Learning rate | scalar ($5 \times 10^{-4}$) | §13 |
| $\mathcal{L}$ | Cross-entropy loss | scalar | §13 |
| $J_C(\phi)$ | Jacobian of capability w.r.t. parameters | $\mathbb{R}^{10 \times p_\phi}$ | §14 |
| $\delta\theta$ | Capability angular displacement | degrees (impl.) | §15 |
| $Q$ | Newman-Girvan modularity | scalar | §16a |
| $W_{i+j}$ | Merged expert parameters | $\mathbb{R}^{p_W}$ | §16b |

---

## CONSISTENCY AUDIT

| Convention | Symbol | Verified? |
|---|---|---|
| Capability vector | $C$ | [VERIFIED] |
| Intervention/environment vector | $\tau$ | [VERIFIED] |
| Model parameters | $\phi$ | [VERIFIED] |
| Capability angular displacement | $\delta\theta$ or $\Delta\theta$ | [VERIFIED] |
| Layer index | $l$ | [VERIFIED] |
| Training checkpoint | $t$ | [VERIFIED] |
| Embedding dimension | $q$ | [VERIFIED] |
| Capability-space dimension | $d = 10$ | [VERIFIED] |

---

## ITEMS REQUIRING MANUAL AUDIT

1. **§15 — Unit Convention:** The `angle_between` function returns **degrees** (`np.degrees(np.arccos(cos_val))`), and the raw parquet column `delta_theta` stores values in degrees. The paper must explicitly choose and label the unit. If radians are preferred in the LaTeX, a conversion note must accompany any table reproducing the raw data.
