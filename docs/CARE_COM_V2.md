# CARE-COM v2.0: Adaptive Capability-Guided Expert Compression

## Motivation
The original CARE-COM v1 framework demonstrated that static capability geometry (the functional output behavior of experts on a semantic calibration probe) acts as a powerful predictor of merge-induced functional damage, substantially outperforming local weight or activation descriptors. However, v1 relies solely on the initial capability geometry to determine the entire sequence of compression operations.

Recent diagnostic experiments (7A, 7B, 7C) ruled out fine-grained neuron-level interactions as the primary driver of residual prediction errors, pointing towards complex interference during weight merging or downstream routing dynamics. 

CARE-COM v2.0 pivots the conceptual role of capability geometry:
- **v1**: Static capability geometry -> direct compression decision.
- **v2.0**: Capability geometry -> candidate generation -> **actual functional evaluation** -> adaptive decision -> recompute state.

By iteratively exploring the functional consequences of candidate prunings in real-time, v2.0 tests whether adaptive capability-aware selection outperforms static geometry.

*Explicit Constraints:*
- **v2.0 DOES NOT use RL, learned policies, or neural network controllers.**
- **v2.0 DOES NOT use neuron-level interaction mechanisms.**
- The initial formulation strictly isolates the *selection* mechanism by relying on pruning and router redistribution rather than weight merging.

## Algorithm Formulations

### 1. Capability Extraction & Redundancy
At each step $t$, the current experts $E_t = \{e_1, ..., e_n\}$ are evaluated using the standard CARE semantic calibration probe to yield $C_i^{(t)} \in \mathbb{R}^{10}$. 
Pairwise distances are defined as:
$$d_C(i, j) = \|C_i^{(t)} - C_j^{(t)}\|_2$$
The nearest neighbor for expert $i$ is:
$$nn(i) = \arg\min_{j \neq i} d_C(i, j)$$

### 2. Candidate Generation & Importance Weighting
To bound computational complexity, we construct a candidate pool of size $K$ (default 8). 
Experts are prioritized for pruning based on redundancy (small distance to nearest neighbor) and optionally down-weighted by their usage importance $u_i$.

The priority score is defined as:
$$score_i = d_C(i, nn(i)) \times (u_i + \epsilon)$$
*Note: A mathematical correction was applied to the specification. The prompt suggested $d_C / (u_i + \epsilon)$, which would counterproductively favor pruning highly-used experts. The implemented $score_i$ uses multiplication so that higher usage increases the score, making the expert harder to select as a pruning candidate.*

The candidates are the $K$ experts with the smallest $score_i$.

### 3. Functional Evaluation & Adaptive Objective
For each candidate $i$ in the pool:
1. Temporarily disable expert $i$.
2. Redistribute its router probability mass (defaulting to the next best routing choice based on model logits, acting as a proxy for nearest-capability if exact top-K probabilities cannot be safely overridden).
3. Evaluate the output KL divergence on the calibration set:
   $$D_i = D_{KL}(P_{base} \parallel P_{candidate})$$

The final selection is the candidate that minimizes actual functional damage:
$$\text{Selected Expert} = \arg\min_{i \in candidates} D_i$$

### 4. Capability-Aware Router Redistribution
When expert $i$ is pruned, its routing mass is redistributed exactly to its capability nearest retained neighbor $j$, but **only** if the native router would have actually selected $i$.
- **Exact Conditional Redistribution Algorithm:** 
  1. The `CapabilityAwareTopKRouter` computes the raw softmax probabilities $p$.
  2. It determines the original top-K selection indices.
  3. For every token where the removed expert $i$ is in the original top-K set, it transfers the mass directly:
     $$p'_j = p_j + p_i$$
     $$p'_i = 0$$
  4. For tokens where $i$ is NOT in the original top-K set, the routing distribution is left unchanged.
- This guarantees that tokens intended for expert $i$ are absorbed entirely by expert $j$ in capability-space mapping, bypassing the standard Top-K fall-back logic, while avoiding transferring non-selected probability mass that could silently alter unselected tokens' dynamics.
- After conditional mass transfer, the Top-K routing selection and final weight normalization occur normally, preserving exactly $K$ active experts per token and the integrity of the sparse forward pass.

## Expected Outputs
The driver outputs a structured JSON trace of the compression loop, detailing for each iteration:
- Candidate pool sizes and distance metrics
- Empirical KL damage for each candidate
- The final selected expert, destination expert, and evaluation times
This trace enables downstream analysis comparing v2.0's decisions against v1's static predictions.

## Configuration & Complexity
- **`candidate_pool_size`**: Limits the number of forward passes per compression step.
- **`candidate_scoring`**: Ablation hooks for `capability_only`, `usage_only`, or `capability_plus_usage`.
- **`adaptive_recompute`**: Ablation hook to test whether capability state recomputation is necessary.
- **`capability_redistribution`**: Ablation hook to test exact mass transfer.
The total cost to remove $N$ experts is bounded by $N \times K \times \text{cost}(\text{forward passes on calibration set})$. No gradients are computed.

## Architectural Constraints
- **Routing Isolation:** The capability-aware redistribution required replacing the top-k selection logic directly. To achieve this without rewriting standard `OlmoeSparseMoeBlock`, v2.0 intercepts the linear gate weights and executes `CapabilityAwareTopKRouter` natively as a monkey-patched replacement during evaluation. This correctly overrides semantics for experimental analysis while leaving the base HuggingFace codebase untouched.
