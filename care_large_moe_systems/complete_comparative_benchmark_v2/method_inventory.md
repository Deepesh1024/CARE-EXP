# Complete MoE Compression Method Inventory (Qwen3-30B-A3B)

This inventory documents all evaluated compression methods, their academic provenance, upstream implementation source, algorithmic categorization, merge/pruning operators, and adaptation status for `Qwen/Qwen3-30B-A3B`.

---

## 1. Method Provenance & Algorithmic Specifications

| Method | Type | Source Paper & Venue | Upstream Repository / Commit | Operator & Mechanism | Target Experts Evaluated | Qwen3-30B Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense Baseline** | Baseline | Qwen3 Technical Report | Hugging Face: `Qwen/Qwen3-30B-A3B` | N/A (128 initial experts per layer, top-8 active) | 128 | **Validated (Pristine)** |
| **CARE-Adaptive** | Merging | CARE-COM (ICLR 2027 Submission) | In-repo: `audit/QWEN3_CARE_COM_RESULTS.json` | Geometry-guided candidate nomination; pairwise arithmetic averaging $\frac{1}{2}(\mathbf{W}_i + \mathbf{W}_j)$ across dimension 0; iterative subspace recalculation | 112, 96, 80, 64, 60 | **Validated (Exp 13/14 Trace Replay)** |
| **REAP** | Pruning | REAP: Router-guided Expert Activation Pruning | In-repo: `benchmark/OLMoE-1B-7B/external/reap` | Router activation profiling across calibration text; global mean routing score; sequential elimination of lowest-frequency experts | 112, 96, 80, 64, 60 | **Validated (Exp 13/14 Baselines)** |
| **Sub-MoE** | Clustering | Sub-MoE: Saliency-guided Expert Clustering | In-repo: `benchmark/OLMoE-1B-7B/external/reap` | Saliency-weighted clustering; cluster centroids retained without subspace regularization | 112, 96, 80, 64, 60 | **Validated (Exp 13/14 Baselines)** |
| **Random Merging** | Merging | Baseline Control | In-repo: `run_qwen3_baselines.py` | Stochastic uniform random pairing without replacement; pairwise arithmetic mean across dimension 0 | 112, 96, 80, 64, 60 | **Validated (Exp 13/14 Baselines)** |
| **HC-SMoE** | Clustering | HC-SMoE (ICML 2025) | `moe-expert-compress` (commit `bd77eba`) | Functional representation proxy from weight row/col means; pairwise Euclidean distances; agglomerative clustering (average linkage) to $K$ clusters; dominant expert selection; cluster parameter merging | 112, 96, 80, 64, 60 | **Faithfully Implemented & Validated** |
| **MC-SMoE / M-SMoE** | Merging | M-SMoE (UNITES-Lab, 2024) | `moe-expert-compress` / UNITES-Lab | Router-guided core selection (top-$K$ by usage frequency); router weight cosine similarity clustering; frequency-weighted parameter averaging | 112, 96, 80, 64, 60 | **Faithfully Implemented & Validated** |

---

## 2. Adaptation Details for Qwen3-30B-A3B

### Architectural Characteristics
In `transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock`:
- All expert MLPs are concatenated along dimension 0 in two 3D tensors:
  - `block.experts.gate_up_proj`: Shape `[num_experts, 1536, 2048]` (gate and up projections concatenated along dim 1).
  - `block.experts.down_proj`: Shape `[num_experts, 2048, 768]`.
- Router gating module:
  - `block.gate.weight`: Shape `[num_experts, 2048]`.
- Top-K Router:
  - `block.gate.top_k`: Dynamic routing budget ($k \in [1, E]$).
  - `block.gate.norm_topk_prob = True`: Guarantees probability renormalization to sum to 1.0.

### Adaptations Documented
1. **HC-SMoE:**
   - Functional expert representation vector:
     $$\mathbf{r}_e = \text{Normalize}\left(\frac{\text{mean}(\mathbf{W}_{\text{gate\_up}, e}, \text{dim}=1) + \text{mean}(\mathbf{W}_{\text{down}, e}, \text{dim}=2)}{2}\right) \in \mathbb{R}^{2048}$$
   - Pairwise distance matrix computed on CPU via PyTorch vectorization.
   - Agglomerative clustering assigns cluster indices $0 \dots K-1$.
   - Slices and packs 3D parameters into `[target_k, 1536, 2048]` and `[target_k, 2048, 768]`.
   - Router weights packed into `[target_k, 2048]`.
2. **MC-SMoE:**
   - Expert usage frequency computed via router projection norms.
   - Cosine similarity computed directly from normalized router vectors:
     $$\mathbf{S} = \bar{\mathbf{W}}_R \bar{\mathbf{W}}_R^T \in \mathbb{R}^{128 \times 128}$$
   - Top-$K$ core centroids identified; non-core experts assigned to highest-similarity core.
   - Slices and packs 3D parameters via usage-weighted combination.
