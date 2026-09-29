# REAP JetMoE-8B Design Document

## 1. Official REAP Scoring Procedure
REAP (Router-weighted Expert Activation Pruning) calculates a saliency score for each expert based on its activation norm and its router gate weight. The official formula is:
$$S_j = \frac{1}{N_j} \sum_{t \in \mathcal{T}_j} \left( g_j(t) \cdot \|f_j(t)\|_2 \right)$$
Where:
- $S_j$: The saliency score for expert $j$.
- $g_j(t)$: The router gate weight for token $t$.
- $\|f_j(t)\|_2$: The L2 norm of the output activation of expert $j$ for token $t$.
- $\mathcal{T}_j$: The set of tokens routed to expert $j$.
- $N_j$: The total number of tokens routed to expert $j$ (size of $\mathcal{T}_j$).

## 2. Required Inputs
- Original JetMoE-8B model with batched tensors (`jetmoe/jetmoe-8b`).
- Tokenized calibration dataset.
- Forward pass hooks to capture both the router logits and the output of the `JetMoeParallelExperts`.

## 3. Calibration Data Requirements
- **Dataset:** WikiText-2 (train split), to perfectly match the calibration conditions used by Sub-MoE and CARE Adaptive.
- **Tokens/Sequence:** 64 sequences of length 512, identical to the Phase 5/8 scripts.
- **Fairness:** The test split will strictly NOT be used for calibration or expert selection.

## 4. Expert-Selection Procedure
1. Run calibration dataset through the model.
2. For each MLP layer, collect $S_j$ for all 8 experts.
3. Rank the experts by $S_j$.
4. Select the $N$ experts with the lowest $S_j$ scores to be pruned (where $N \in \{1, 2, 4\}$ depending on the compression target).

## 5. Pruning Procedure & Router Handling
Unlike expert merging (which leaves the tensor shape intact and just overwrites weights), REAP is a physical pruning method.
To prune an expert in JetMoE:
- Identify the retained expert indices (e.g., if expert 2 is pruned, retained are `[0, 1, 3, 4, 5, 6, 7]`).
- Physically slice the batched MLP tensors:
  - `mlp.experts.input_linear.weight` becomes `[7, 11264, 2048]`.
  - `mlp.experts.output_linear.weight` becomes `[7, 2048, 5632]`.
- Update the router:
  - Slicing `mlp.router.weight` from `[8, dim]` to `[7, dim]`.
  - Setting `mlp.router.num_experts = 7`.
- Update `model.config.num_experts = 7` (since the compression is applied uniformly across all layers, the global config can be safely updated).

## 6. Incompatibilities with JetMoE & Required Adapters
- The official REAP codebase supports several architectures but does not have a native adapter for JetMoE's specific batched tensor format (`JetMoeParallelExperts`) out of the box.
- Therefore, we will implement a custom JetMoE-compatible adapter `phase10_reap_baseline.py` that hooks into JetMoE's `mlp.router` and `mlp.experts` to collect the statistics during calibration, and then dynamically slices the tensors in memory before saving the pruned models.
- Since JetMoE also contains MoA (Mixture of Attention) experts, our adapter will strictly target `model.model.layers[i].mlp` and ignore the MoA layers to match the constraints of the benchmark.

## 7. Saving the Compressed Models
- The physically pruned models will be saved to disk at:
  - `benchmark_results/JetMoE-8B/jetmoe_reap_7`
  - `benchmark_results/JetMoE-8B/jetmoe_reap_6`
  - `benchmark_results/JetMoE-8B/jetmoe_reap_4`
