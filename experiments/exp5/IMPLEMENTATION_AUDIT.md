# EXPERIMENT 5 IMPLEMENTATION AUDIT (UPDATED)

## RESOLVED

### A. Exact MoE Checkpoint
`allenai/OLMoE-1B-7B-0924` (based on prior experiment configs).

### B. Expert Architecture
**RESOLVED.** The exact structure is `OlmoeExperts`. The experts are batched in dimension 0 for tensors `gate_up_proj` (shape `[64, 2048, 2048]`) and `down_proj` (shape `[64, 2048, 1024]`). See `architecture_inspection.md`.

### C. Router Architecture
**RESOLVED.** The router is `OlmoeTopKRouter` with a single `weight` matrix (shape `[64, 2048]`). Logits are calculated linearly. See `architecture_inspection.md`.

### D. Calibration Dataset
`Salesforce/wikitext` (subset: `wikitext-2-raw-v1`, `train` split, 512 sequence length, 98 sequences = 50,176 tokens).

### F. Oracle KL Independence
**RESOLVED.** See `oracle_dependency_audit.md`. The CARE features, representations, and decisions can be constructed purely from the model and calibration data. Oracle KL is only used for offline training of the predictor (already done in Exp 4) and post-hoc validation.

### J. Exact Merge Operator
**RESOLVED.** The operator averages slices in dimension 0 for `gate_up_proj` and `down_proj` in `OlmoeExperts`, then removes the redundant expert slice. See `merge_operator_spec.md`.

### K. Exact Router Update
**RESOLVED.** The update averages slices in dimension 0 of the router `weight` matrix, preserving logit distribution. See `router_merge_spec.md`.

### L. Exact Conflict-Resolution Procedure
**RESOLVED.** Sort candidates by score, traverse in order, accept if neither expert is already selected in the current stage. Stop when budget is reached.

### M. One-Shot Procedure
**RESOLVED.** Original 64-expert model -> construct representation once -> produce ranking -> perform predefined compression trajectory.

### N. Iterative Procedure
**RESOLVED.** Recompute current state after every merge stage (64 -> 63 -> 62 ... -> target).

### O. Exact Compression Levels
**RESOLVED.** 64, 56, 48, 40, 32, 24, 16.


## VALIDATED FROM EXISTING CODE

### E. Exact Statistics Constituting the CARE Representation
Weight_Distance, Weight_Cosine, Activation_Similarity, Output_Similarity, Routing_Similarity, Usage_Frequency, Jaccard_Overlap, Usage_Asymmetry, Routing_JSD_Proxy, Routing_NPMI_Proxy, Specialization_Diff.

### G. Exact CARE_GEO Algorithm
Pairwise Euclidean distance between experts in the $q=4$ MDS embedding of functional statistics. Shortest distances are ranked highest for merging.

### H. Exact CARE_COM Algorithm
The trained XGBoost local predictor from Exp 4 taking 11 local features + Geometry_Distance as input, outputting predicted Oracle KL damage. Smallest predicted damage is ranked highest.


## REQUIRES RESEARCH DECISION

### P. Benchmarks Used
**REQUIRES DECISION.** See `benchmark_audit.md`. The repository currently contains no benchmark infrastructure. We recommend adding Wikitext Perplexity (Required) and standard NLP benchmarks like MMLU or ARC (Recommended) to evaluate the compressed models properly.

### R. Exact Statistical Comparison Procedure
**REQUIRES DECISION.** See `statistical_protocol.md`. The general framework is established, but it must be approved before moving forward, particularly regarding multiple-comparison handling and practical significance thresholds.


## BLOCKING

### Q. Evaluation-Noise Floor
**BLOCKED.** The evaluation noise floor script (`evaluate_noise_floor.py`) cannot be built and run on the uncompressed model until the Benchmark Suite (P) is decided. We cannot proceed with `evaluation_noise_floor.json` generation until the evaluation method is frozen.

### I. Exact Conventional Baseline Algorithms
**BLOCKED (Partially).** 
- Parameter Similarity metric needs freezing (e.g. Euclidean on flattened tensors).
- Local predictor baseline needs freezing (XGBoost from Exp 4 without Geometry).
- Usage baseline logic is frozen, but conflict resolution orientation needs explicit confirmation.

---
**CONCLUSION:** The prerequisite phase is incomplete. The actual compression implementation remains blocked until the benchmark suite and noise floor generation are resolved.
