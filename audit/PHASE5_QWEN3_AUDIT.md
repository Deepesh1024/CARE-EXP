# Phase 5 Implementation Audit for Qwen3-30B-A3B

This audit maps the functional requirements of the authoritative CARE-COM benchmark (as used in OLMoE experiments) to the specific architecture of Qwen3-30B-A3B to ensure strict scientific rigor.

## 1. Capability Representation (Residual-Stream Probe)
**Authoritative OLMoE Implementation:**
- Passes hidden states into individual experts in a `ModuleList`.
- Computes `out_e = down_proj(act(gate_proj(x)) * up_proj(x))`.
- Calculates capability as the L2 norm of `out_e` per token, masked and averaged.

**Qwen3 Implementation Mapping:**
- Qwen3 experts are fused 3D tensors: `gate_up_proj` and `down_proj`.
- For each `expert_idx`, we will slice the tensor to evaluate it independently:
  ```python
  gate, up = F.linear(x, block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
  out_e = F.linear(act_fn(gate) * up, block.experts.down_proj[expert_idx])
  # Apply attention mask and compute L2 norm identical to OLMoE
  mask_t = mask.unsqueeze(-1)
  out_e = out_e * mask_t
  norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
  ```
- This ensures the exact same residual-stream impact representation.

## 2. Candidate Generation (CARE Geometry)
**Authoritative OLMoE Implementation:**
- Concatenates layer-normalized vectors into a global capability vector.
- Computes pairwise L2 distances in global capability space.
- Selects top-K pairs with minimal distance.

**Qwen3 Implementation Mapping:**
- We will fully reuse `compute_global_capability_vectors` and `pairwise_capability_distances` from `experiments/care_com_v21/capability.py`.
- No proxies (like router weight distances or arbitrary heuristics) will be used.

## 3. Physical Expert Removal & Router Resizing
**Authoritative OLMoE Implementation:**
- Merges weights.
- `block.experts.pop(j)` to remove expert.
- Updates router weight by keeping all rows except `j`.

**Qwen3 Implementation Mapping:**
- Because Qwen3 uses fused `[num_experts, ...]` tensors rather than a `ModuleList`, `.pop()` is impossible.
- We will explicitly tensor-slice all parameter tensors to exclude index `j`:
  ```python
  keep_indices = torch.tensor([k for k in range(n) if k != j], device=device)
  layer.experts.gate_up_proj = nn.Parameter(layer.experts.gate_up_proj.data[keep_indices])
  layer.experts.down_proj = nn.Parameter(layer.experts.down_proj.data[keep_indices])
  layer.gate.weight = nn.Parameter(layer.gate.weight.data[keep_indices])
  layer.experts.num_experts -= 1
  layer.gate.num_experts -= 1
  ```
- **Verification**: This definitively eliminates dead expert slots. The parameters strictly shrink from `128` to `127` in memory.

## 4. KL Evaluation (Damage Assessment)
**Authoritative OLMoE Implementation:**
- Stores a baseline probability distribution over a calibration dataset prior to any compression.
- Evaluates marginal KL divergence of the modified model against the baseline.

**Qwen3 Implementation Mapping:**
- We will import and directly use `compute_calibration_distributions` and `evaluate_marginal_kl` from `benchmark.core.evaluate`.
- The evaluation will compare the physically merged Qwen3 model to the exact pre-intervention baseline probabilities.

## 5. Merging Reversibility
**Authoritative OLMoE Implementation:**
- Uses `copy.deepcopy` on the layer's `state_dict()` and restores it if the candidate is rejected.

**Qwen3 Implementation Mapping:**
- Standard `load_state_dict()` fails if the parameter shapes have changed (e.g., `127` back to `128`).
- We will ensure full structural reversibility by re-initializing the parameters to their original dimensions before loading:
  ```python
  layer.experts.gate_up_proj = nn.Parameter(torch.empty_like(original_state_dict["experts.gate_up_proj"]))
  layer.experts.down_proj = nn.Parameter(torch.empty_like(original_state_dict["experts.down_proj"]))
  layer.gate.weight = nn.Parameter(torch.empty_like(original_state_dict["gate.weight"]))
  layer.load_state_dict(original_state_dict)
  ```
- **Verification**: Our Phase 5 mechanical pilot confirmed this successfully restores the model to the original weights without generating NaN outputs. 

## Conclusion
The pilot demonstrated that the mechanical requirements are achievable. To run the full benchmark, we will instantiate a specific Qwen3 `PhysicalMergeEngine` that overrides the OLMoE manipulation methods but routes all higher-order logic (Capability computation, KL evaluation, Ranking) through the authoritative CARE-COM functions.
