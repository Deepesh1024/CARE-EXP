# Phase 6: Baseline Fidelity Audit & Algorithmic Deviations

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 18, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifact Location:** [`audit_repair_pass/pre_review_sprint/BASELINE_FIDELITY.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/pre_review_sprint/BASELINE_FIDELITY.md)  

---

## 1. Executive Summary & Classification Standard

This audit rigorously inspects the baseline implementations used across both the `OLMoE-1B-7B-0924` (Table 1, Table 3) and `Qwen3-30B-A3B` benchmark tables. Every implementation is classified into one of three strict scientific tiers:
- **Tier A: Faithful Reproduction (Supported by direct code and mathematical evidence)**
- **Tier B: Named Approximation (Documented deviations; valid comparative proxy)**
- **Tier C: Invalid / Incomplete Result (Excluded from primary comparative claims)**

---

## 2. Method-by-Method Audit

### 2.1. HC-SMoE (Hierarchical Clustering for Sparsely-Activated MoE)
- **Original Source:** Chen et al., ICML 2025 (`https://github.com/wazenmai/HC-SMoE`)
- **Published Algorithm:**
  1. Profile token-level output representations $O_i = E_i(X)$ across calibration sequences.
  2. Compute pairwise cosine distance $D(i, j) = 1 - \frac{\langle O_i, O_j \rangle}{\|O_i\| \|O_j\|}$.
  3. Hierarchical Agglomerative Clustering (HAC) using average linkage.
  4. Parameter averaging within clusters.
- **Repository Implementation (`qwen3_clustering_engines.py` / `benchmark/OLMoE-1B-7B/`):**
  - Uses the exact **Lance-Williams average-linkage recurrence formula**:
    $$D(c_i \cup c_j, k) = \frac{n_i D(c_i, k) + n_j D(c_j, k)}{n_i + n_j}$$
    reducing complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(N^2)$.
  - **Deviation:** In Qwen3-30B (48 layers), caching token forward activations across 128 experts would exceed 120 GB of VRAM/RAM. The implementation substituted empirical activations with a **weight-space proxy representation**:
    $$\mathbf{r}_i = \text{Normalize}\left(\frac{\text{row\_mean}(W_{\text{gate\_up}, i}) + \text{col\_mean}(W_{\text{down}, i})}{2}\right)$$
- **Classification:** **TIER B — Named Approximation**  
  **Designated Label:** `HC-SMoE (Lance-Williams HAC with Weight-Proxy Output Representations)`

---

### 2.2. MC-SMoE / M-SMoE (Merge, Then Compress)
- **Original Source:** Li et al., ICLR 2024 (`https://github.com/UNITES-Lab/MC-SMoE`)
- **Published Algorithm:**
  1. Neuron Permutation Alignment: Bipartite matching to align hidden channels.
  2. Routing-Guided Core Selection: Sort experts by routing frequency $f_i$; top-$k$ become core centroids.
  3. Non-core assignment: Assign non-core experts to the nearest core centroid via router cosine similarity.
  4. Frequency-weighted parameter fusion.
  5. C-SMoE: Post-merge low-rank SVD compression of weights.
- **Repository Implementation (`qwen3_clustering_engines.py`):**
  - Faithfully reproduces Stage-1 M-SMoE: Core selection based on routing frequency/norms and router weight cosine similarity assignment.
  - **Deviations:**
    1. *Neuron Permutation Omitted:* Fused gated SiLU activations (`gate_up_proj`) break linear channel permutations without retraining.
    2. *C-SMoE SVD Omitted:* The benchmark targets expert pool reduction ($E \in \{112, 96, 80, 60\}$), not intra-expert low-rank compression.
- **Classification:** **TIER B — Named Approximation**  
  **Designated Label:** `M-SMoE (Routing-Guided Core Expert Merging)`

---

### 2.3. Sub-MoE (Subspace Expert Merging)
- **Original Source:** Li et al., AAAI 2026 / arXiv 2025 (`https://arxiv.org/abs/2506.23266`)
- **Published Algorithm:**
  1. Group experts by Characteristic Activation (CA) cosine similarity.
  2. Experts Union Decomposition: Compute joint SVD on concatenated weights $[W_1; \dots; W_m] = U \Sigma V^T$.
  3. Retain dominant left singular vectors $U$ as shared subspace basis.
  4. Subspace parameter blending on right singular vectors $V$.
- **Repository Implementation (`run_qwen3_baselines.py` / `run_exp14_baselines_continuation.py`):**
  - Faithfully extracts Characteristic Activation (CA) vectors from `gate_up_proj` row means.
  - Clusters experts via agglomerative clustering on CA cosine similarity.
  - **Critical Deviation:** Merges clustered experts via **direct parameter averaging** $(W_i + W_j)/2$ rather than joint SVD subspace projection.
  - **Impact:** Direct parameter averaging of disparate experts causes severe destructive interference at aggressive compression (PPL collapses to 90.72 at 80E and 203.65 at 60E on Qwen3-30B).
- **Classification:** **TIER B — Named Approximation**  
  **Designated Label:** `Sub-MoE Proxy (CA-Cluster Parameter Averaging)`

---

### 2.4. REAP (Router-guided Activation Pruning)
- **Original Source:** Lasby et al., ICLR 2026 (`https://arxiv.org/abs/2510.13999`)
- **Published Algorithm:**
  1. Profile token routing frequencies across calibration sequences.
  2. Compute global router-weighted activation saliency per expert.
  3. Permanently drop (prune) the lowest-saliency experts without altering surviving weights.
- **Repository Implementation (`run_qwen3_baselines.py` / `benchmark/OLMoE-1B-7B/methods/reap_family.py`):**
  - Exactly implements calibration profiling, saliency ranking, and parameter tensor slicing.
- **Classification:** **TIER A — Faithful Reproduction**  
  **Designated Label:** `REAP (Router-guided Expert Activation Pruning)`  
  *Note: Categorized strictly as a pruning baseline and excluded from merging-specific primary tables.*

---

### 2.5. Historical Phi-3.5-MoE Benchmark Run (Invalidated)
- **Historical Implementation:** Early benchmark runs on Phi-3.5-MoE (`submoe`, `rw_l2`, `parameter`, `care_static`) did not physically delete merged expert tensors from memory or model forward graphs, artificially retaining uncompressed performance.
- **Classification:** **TIER C — Invalid / Incomplete Result**  
  **Remediation:** Formally excluded from all scientific comparison tables in `full_report.md:L432`.

---

## 3. Systematic Algorithmic Comparison Matrix

| Algorithmic Dimension | CARE-Adaptive | HC-SMoE (Repo) | M-SMoE (Repo) | Sub-MoE Proxy (Repo) | REAP |
|---|---|---|---|---|---|
| **Paradigm** | Dynamic Pairwise Merge | Hierarchical Agglomerative | Routing-Guided Core | CA Centroid Clustering | Activation Pruning |
| **Selection Criterion** | Capability Distance + Marginal KL | Weight-Space Output Proxy | Router Weight Cosine Sim | Characteristic Activation Cosine | Router Activation Saliency |
| **Weight Operator** | Parameter Average $(W_i+W_j)/2$ | Cluster Mean Average | Frequency-Weighted Average | Parameter Average $(W_i+W_j)/2$ | Expert Dropped (No Merge) |
| **Joint SVD / Subspace** | No | No | No (C-SMoE omitted) | No (SVD omitted) | N/A |
| **Router Adaptation** | Row Average $(W_{r, i}+W_{r, j})/2$ | Cluster Sum $\sum W_{r, m}$ | Weighted Sum $\sum w_m W_{r, m}$ | Row Average $(W_{r, i}+W_{r, j})/2$ | Sliced Parameter Tensor |
| **Fidelity Tier** | **Tier A (Proposed Method)** | **Tier B (Named Approx)** | **Tier B (Named Approx)** | **Tier B (Named Approx)** | **Tier A (Faithful Reproduction)** |

---

## 4. Manuscript Action Items

1. **Update Baseline Method Names:**  
   In all revised text, tables, and captions, rename `Sub-MoE` to `Sub-MoE Proxy (CA-Cluster Merging)` and `MC-SMoE` to `M-SMoE (Core Selection & Merging)`.
2. **Qualify Comparative Claims:**  
   Do not claim universal algorithmic superiority over the published papers of Chen et al. (2025), Li et al. (2024), or Li et al. (2026). State explicitly that comparisons reflect the specified open-source and computationally tractable implementations evaluated under a standardized sequential benchmark.
