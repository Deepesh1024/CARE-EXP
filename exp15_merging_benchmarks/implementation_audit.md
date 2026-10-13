# Implementation Audit: HC-SMoE and MC-SMoE Benchmark (Exp 15)

## 1. Executive Summary & Objective

This audit establishes the formal mathematical, algorithmic, and architectural foundation for **Experiment 15: HC-SMoE and MC-SMoE Benchmarks** on `JetMoE-8B` and `Phi-3.5-MoE`. 

We evaluate two state-of-the-art SMoE merging and consolidation methods:
1. **HC-SMoE** (Hierarchical Clustering for SMoE, ICML 2025)
2. **MC-SMoE** (Merge, Then Compress SMoE, UNITES Lab, 2024)

These methods are benchmarked against existing validated reference baselines (**CARE-Adaptive**, **REAP**, **Sub-MoE**, and **Random**) without re-running or recomputing any prior artifacts.

---

## 2. Official Repositories & Provenance

| Method | Official Repository | Commit Hash | Upstream Date | Canonical Architectures |
| :--- | :--- | :--- | :--- | :--- |
| **HC-SMoE** | `https://github.com/wazenmai/HC-SMoE` | `bd77eba1b55ca58606a47e2d0807a24218da73c3` | May 24, 2025 | Mixtral-8x7B, Qwen1.5-MoE-A2.7B |
| **MC-SMoE** | `https://github.com/UNITES-Lab/MC-SMoE` | `99b400f0d5b3be0015ebec4a5a80a7821be1aebd` | Jun 20, 2025 | SwitchTransformers, FSGPT, Mixtral-8x7B |

---

## 3. Algorithm Deconstruction & Mathematical Comparison

### 3.1 HC-SMoE (Hierarchical Clustering SMoE)
* **Core Philosophy**: Task-agnostic, retraining-free merging based on **functional expert output similarity**, completely bypassing router bias.
* **Clustering Operator**:
  * For input tokens $X$, compute expert forward activations $E_i(X) \in \mathbb{R}^{T \times D}$.
  * Pairwise distance $D(i, j) = 1 - \text{CosineSimilarity}(\bar{E}_i(X), \bar{E}_j(X))$ or Euclidean distance $\|E_i(X) - E_j(X)\|_2$.
  * Agglomerative hierarchical clustering with **Ward linkage** or **Average linkage** down to target cluster count $K$.
* **Center (Dominant) Expert Selection**:
  * Within each cluster $C_k$, the expert closest to the cluster centroid in representation space is designated as the dominant/core expert.
* **Weight Merging Operator**:
  * Within cluster $C_k$, weights are combined via frequency-weighted or uniform averaging:
    $$W_{merged}^{(k)} = \sum_{i \in C_k} \alpha_i W_i, \quad \alpha_i = \frac{f_i}{\sum_{j \in C_k} f_j}$$
* **Router Handling**:
  * In the official repository, HC-SMoE tests both router-preserving soft binding and router-row merging:
    $$R_{merged}^{(k)} = \sum_{i \in C_k} \alpha_i R_i$$

### 3.2 MC-SMoE (Merge, Then Compress SMoE)
* **Core Philosophy**: Routing-guided clustering using activation frequency to designate dominant "core" experts, followed by usage-weighted parameter averaging, and optional low-rank/sparse decomposition (LoSparse).
* **Two-Stage Pipeline Delineation**:
  * **Stage 1 (M-SMoE — Merging Only)**: 
    1. Neuron permutation alignment (optional/expensive Hungarian matching).
    2. Dominant expert selection: sort experts by activation frequency $f_i$; top-$K$ experts become cluster seeds.
    3. Non-core assignment: each remaining expert $j > K$ is assigned to the core expert $i \le K$ with highest router-logit/weight cosine similarity.
    4. Merging:
       $$W_{merged}^{(i)} = \frac{f_i W_i + \sum_{j \in C_i} f_j W_j}{f_i + \sum_{j \in C_i} f_j}$$
  * **Stage 2 (C-SMoE — LoSparse Compression)**:
    * Decomposes merged expert matrices into Low-Rank ($L$) + Sparse ($S$) components ($W = L + S$) using SVD and magnitude pruning.
  * *Critical Audit Rule:* To ensure a scientifically fair comparison with CARE-Adaptive and Sub-MoE, we evaluate **Stage 1 (M-SMoE Merging-Only)** as the primary baseline, and explicitly label whether LoSparse is applied.

---

## 4. Architectural Compatibility & Required Adaptations

Neither upstream repository supports `JetMoE-8B` or `Phi-3.5-MoE-instruct` out of the box. An adapted implementation is required:

### 4.1 Target 1: JetMoE-8B (`jetmoe/jetmoe-8b`)
* **Layer Composition**: 24 Transformer layers.
* **MoE Structure**: Dual MoE — Mixture of Attention (MoA) and Mixture of MLP (MoE).
* **Expert Tensor Representation**: Batched 3D tensors:
  * `input_linear.weight`: `[8, 11264, 2048]` (fused SwiGLU W1 + W3)
  * `output_linear.weight`: `[8, 2048, 5632]` (W2)
  * `router.layer.weight`: `[8, 2048]`
* **Required Adaptation**:
  * Slicing along dimension 0 rather than indexing a `nn.ModuleList`.
  * Merging target: MoE MLP layers ($8 \to 7 \to 6 \to 4$).
  * Physical dimension reduction: constructing new tensors of shape `[K, 11264, 2048]` and updating `config.num_local_experts = K`.

### 4.2 Target 2: Phi-3.5-MoE (`microsoft/Phi-3.5-MoE-instruct`)
* **Layer Composition**: 32 Transformer layers.
* **MoE Structure**: 16 experts per layer, top-2 routing.
* **Expert Tensor Representation**: Fused weights:
  * `gate_up_proj`: `[16, 12800, 4096]` (W1 and W3 concatenated)
  * `down_proj`: `[16, 4096, 6400]` (W2)
  * `gate.weight`: `[16, 4096]`
* **Required Adaptation**:
  * Dimension slicing on dim 0.
  * Invariant enforcement: $N_{gate\_up} = N_{down} = N_{gate} = K$.
  * Evaluation checkpoints: $16 \to 14 \to 12 \to 10 \to 8$.

---

## 5. Faithful Reproduction vs. Adaptation Classification

| Aspect | Faithful Upstream Logic Preserved | Necessary Adaptation |
| :--- | :--- | :--- |
| **HC-SMoE** | Output-based distance metric, agglomerative hierarchical clustering, Ward/Average linkage, centroid-based dominant selection. | Adapted tensor slicing for batched MoE (`JetMoE`) and fused projection layouts (`Phi-3.5-MoE`). |
| **MC-SMoE** | Routing frequency dominant selection, router/weight similarity-based cluster assignment, frequency-weighted parameter averaging. | Separated Stage 1 (M-SMoE) from Stage 2 (LoSparse); adapted tensor slicing for parallel batched layouts. |

---

## 6. Reference Baseline Summary (Do Not Recompute)

### JetMoE-8B Reference Results
* **Original Model Baseline PPL**: `7.1258` (subset: `7.5085`)
* **CARE-Adaptive PPL**:
  * 7 experts: `10.17`
  * 6 experts: `15.69`
  * 4 experts: `87.19`
* **REAP PPL**:
  * 7 experts: `8.10`
  * 6 experts: `9.81`
  * 4 experts: `17.48`

### Phi-3.5-MoE Reference Results
* **Original Model Baseline PPL**: `4.57`
* **CARE-Adaptive PPL**:
  * 14 experts: `6.88`
  * 12 experts: `9.24`
  * 10 experts: `13.57`
  * 8 experts: `32.75`
* **REAP PPL**:
  * 14 experts: `6.44`
  * 12 experts: `17.65`
  * 10 experts: `40.33`
  * 8 experts: `125.72`
* **Random PPL**:
  * 14 experts: `8.50`
  * 12 experts: `23.34`
  * 10 experts: `44.74`
  * 8 experts: `71.90`
