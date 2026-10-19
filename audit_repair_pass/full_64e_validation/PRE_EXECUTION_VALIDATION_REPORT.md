# Pre-Execution Validation Report: Full 64-Expert CARE Rematch

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 19, 2026  
**Auditor:** Antigravity Autonomous Experimental Auditor  
**Status:** VALIDATION COMPLETE — EXECUTION FROZEN PENDING USER CONFIRMATION  
**Artifact:** [`audit_repair_pass/full_64e_validation/PRE_EXECUTION_VALIDATION_REPORT.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/full_64e_validation/PRE_EXECUTION_VALIDATION_REPORT.md)  
**Preserved Baseline Manuscript Hash:** `72489204cfea2e51590b8631444eaa85d8f7a074a18359653a745f0ff4550a65` (100% UNMODIFIED)

---

## 1. Exact Cause of the Expert-Index Restriction in `capability.py`

Forensic static and dynamic execution analysis of `experiments/care_com_v21/capability.py` identified **three compounding flaws** responsible for restricting candidate nominations to indices $[0, 15]$:

### Flaw 1 (The Root Mathematical Bug): Layer-Normalization Slicing Confusion
In lines 65–70 of `experiments/care_com_v21/capability.py`:
```python
# layer_capabilities was allocated as shape: [num_layers, num_experts, 10]
# For OLMoE: num_layers = 16, num_experts = 64
for layer_idx in range(num_layers): # layer_idx in [0, 15]
    layer_C = layer_capabilities[:, layer_idx, :]  # <--- CRITICAL BUG
    norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    layer_capabilities[:, layer_idx, :] = layer_C / norms
```
- In `layer_capabilities` (shape `[16, 64, 10]`), dimension 0 is `num_layers` (16) and dimension 1 is `num_experts` (64).
- The author intended to slice each layer across all experts: `layer_capabilities[layer_idx, :, :]`.
- Instead, the author wrote `layer_capabilities[:, layer_idx, :]`. Because `layer_idx` only ranged from $0$ to $15$ (`range(num_layers)`), it **sliced dimension 1 (experts) from index 0 to 15**, treating the 16 layers as 16 experts.
- Consequently, only experts $0 \dots 15$ received normalized unit vectors. Experts $16 \dots 63$ were left completely unnormalized with large, unscaled activation norms, severely distorting Euclidean pairwise distances and excluding them from nearest-neighbor candidate nomination.

### Flaw 2: Fused 3D Tensor Incompatibility
In PyTorch HuggingFace OLMoE (`OlmoeSparseMoeBlock`), the MLP experts are batched into a single 3D module `OlmoeExperts`:
- `gate_up_proj`: `[64, 2 * intermediate_size, hidden_size]`
- `down_proj`: `[64, hidden_size, intermediate_size]`
`OlmoeExperts` does not implement `__len__()` or subscription `block.experts[exp_idx]`. In `capability.py:14`, `len(moe_blocks[0].experts)` throws `TypeError: object of type 'OlmoeExperts' has no len()`. In earlier mock runs or ported Phi-3.5 code (which used 16 unfused experts), `num_experts` defaulted to 16.

### Flaw 3: Static Initialization Instead of Dynamic Tracking
`capability.py` read expert counts once at initialization instead of dynamically accepting `engine.current_num_experts` ($N_t \in [48, 64]$) from `PhysicalMergeEngine`.

---

## 2. Verification of Full Active Expert Nomination Across Re-indexing

In [`audit_repair_pass/full_64e_validation/care_capability_v23.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/full_64e_validation/care_capability_v23.py), all three flaws have been resolved:

1. **Layer Slicing Corrected:** Slices `layer_capabilities[layer_idx, :, :]` across all $N_t$ active experts.
2. **Dual-Architecture Forward Dispatch:** 
   - Fused 3D: Uses `F.linear(x, block.experts.gate_up_proj[exp_idx])` and `F.linear(act, block.experts.down_proj[exp_idx])`.
   - Unfused ModuleList: Uses `exp.gate_proj.weight`, `exp.up_proj.weight`, `exp.down_proj.weight`.
3. **Sequential Merge Dynamic Invariant:**
   Unit test `test_invariant_4_dynamic_tracking_across_sequential_merges` in [`test_candidate_selection_64e.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/full_64e_validation/tests/test_candidate_selection_64e.py) executed 16 sequential physical merges from 64 down to 48 experts. At every step:
   - `C` had exact shape $[N_t, 40]$ (for 4 test layers) or $[N_t, 160]$ (for 16 layers).
   - Every nominated candidate pair $(i, j)$ satisfied $0 \le i < j < N_t$.
   - **Zero index collisions, zero out-of-range indices, and zero stale references occurred.**

---

## 3. Capability Vector Alignment & Stale-Index Elimination

- **Physical Index Alignment:** In `PhysicalMergeEngine.merge_experts(expert_i, expert_j)`, expert $j$ is deleted, and experts $> j$ shift down by 1 index. In `compute_global_capability_vectors_v23`, expert slice `e` in tensor dimension 0 maps 1:1 to index `e` in the active model.
- **Key Invariant Tested:** `test_invariant_5_zero_stale_indices` verified that after merging experts $(10, 20)$, the capability distance dictionary contains exactly 63 keys ($0 \dots 62$) and contains **no reference to index 63**.
- **Transactional Restoration:** Every candidate evaluation wraps in `engine.snapshot(i)` and `engine.restore()`. Restoring Layer 0 Gate tensor was verified via SHA-256 bit-level fingerprinting before and after evaluation.

---

## 4. Controlled Parity Matrix: CARE-Adaptive vs. Random-K=5

To ensure scientific parity, CARE-Adaptive and Random-$K=5$ are strictly matched across all execution dimensions:

| Experimental Parameter | Random-$K=5$ (Audited Baseline) | CARE-Adaptive Rematch (v2.3) | Parity Status |
|---|---|---|:---:|
| **Base Model Checkpoint** | `allenai/OLMoE-1B-7B-0924` | `allenai/OLMoE-1B-7B-0924` | **IDENTICAL** |
| **Model Precision** | `bfloat16` | `bfloat16` | **IDENTICAL** |
| **Initial Physical Experts** | 64 | 64 | **IDENTICAL** |
| **Target Physical Experts** | 48 (16 sequential merges) | 48 (16 sequential merges) | **IDENTICAL** |
| **Intervention Budget ($K$)** | Exactly 5 candidate merges per step | Exactly 5 candidate merges per step | **IDENTICAL** |
| **Total Physical Interventions**| $16 \times 5 = 80$ evaluations | $16 \times 5 = 80$ evaluations | **IDENTICAL** |
| **Paired Random Seeds** | `[101, 202, 303, 404, 505]` | `[101, 202, 303, 404, 505]` | **IDENTICAL** |
| **Calibration Draw Protocol** | `df_full.sample(n=128, random_state=seed)` | `df_full.sample(n=128, random_state=seed)` | **IDENTICAL** |
| **Candidate Scoring Loss** | `evaluate_marginal_kl` on 128 chunks | `evaluate_marginal_kl` on 128 chunks | **IDENTICAL** |
| **Router Combination Rule** | $(W_{r, i} + W_{r, j}) / 2.0$ | $(W_{r, i} + W_{r, j}) / 2.0$ | **IDENTICAL** |
| **Nomination Scope** | All active pairs $0 \le i < j < N_t$ | All active pairs $0 \le i < j < N_t$ | **IDENTICAL** |

---

## 5. Artifact Preservation & Versioning Architecture

To protect experimental integrity:
- **Historical Code:** Preserved unchanged in `experiments/care_com_v21/capability.py` and `benchmark/OLMoE-1B-7B/methods/care_com.py`.
- **Historical Trajectories:** Preserved unchanged in `benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json` (SHA-256: `408f2faddb45d43e...`).
- **Submitted Manuscript:** Preserved unchanged in `audit_repair_pass/final_validation/SUBMITTED_MANUSCRIPT_ICLR2027.pdf` (SHA-256: `72489204cfea...`).
- **Versioned Corrected Implementation:** Isolated in `audit_repair_pass/full_64e_validation/`:
  - Engine: `care_capability_v23.py`
  - Runner: `run_matched_care_64e.py`
  - Unit Tests: `tests/test_candidate_selection_64e.py`
  - Output Trajectories: `audit_repair_pass/full_64e_validation/care_adaptive_64e/`

---

## 6. Verification of the 65,536-Token WikiText-2 Test Protocol

The validation perplexity protocol in `run_matched_care_64e.py` calls `compute_ppl` from `benchmark/OLMoE-1B-7B/core/evaluate.py`:
- **Dataset:** `Salesforce/wikitext` (`wikitext-2-raw-v1`, `test` split).
- **Chunking:** Non-overlapping contiguous sequences of length 512.
- **Evaluation Chunks:** Exactly 128 sequences.
- **Total Token Count:** $128 \times 512 = \mathbf{65,536\text{ tokens}}$.
- **Loss Computation:** Causal cross-entropy with padding masked to -100.
- **Checkpoints Evaluated:** $N \in [64, 60, 56, 48]$ experts.

---

## 7. Review of Decision Thresholds & Experimental Design Match

### A. Temporal Pre-Registration Verification
The decision boundaries below were formally recorded in [`NEXT_EXPERIMENT_PREREGISTRATION.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/random_k5_forensic_audit/NEXT_EXPERIMENT_PREREGISTRATION.md) **prior to executing any runs**:

```
                       Pre-Registered Decision Spectrum
  
  [--- DECISIVE SUCCESS ---]      [-- QUALIFIED MERIT --]       [--- CLAIM FALSIFIED ---]
 10.28 (Base)            22.00   22.00                 26.00   29.81 (Rand-5 Mean)     35.0+
   |-----------------------|-------|---------------------|-------|-----------------------|
   ^ Full 64E CARE beats   ^       ^ Statistically       ^       ^ CARE is no better than
   every random seed with          better than Rand-5,           random search under equal
   p < 0.01 (paired t-test)        but search gap shrinks        budget. Claim rejected.
```

- **Boundary A (Decisive Confirmation):** Mean CARE PPL $\le 22.00$ and $\Delta_s > 0$ for all 5 seeds ($p < 0.01$).
- **Boundary B (Qualified Merit):** Mean CARE PPL $\in (22.00, 26.00]$ and $p < 0.05$.
- **Boundary C (Claim Falsified):** Mean CARE PPL $\ge 28.00$ or $p \ge 0.05$.

### B. Statistical Alignment with Experimental Design
- In the audited validation run, Random-$K=5$ had 5 runs across seeds $\{101, 202, 303, 404, 505\}$ with mean $29.814 \pm 3.436$.
- In this rematch, CARE-Adaptive evaluates the exact same 5 seeds with the exact same calibration subset draws.
- This creates 5 paired observations:
  $$\Delta_s = \text{PPL}_{\text{Random-}K=5}(s) - \text{PPL}_{\text{CARE-Adaptive}}(s)$$
- The hypothesis test is a **true paired Student's $t$-test ($df = 4$)**:
  $$t = \frac{\bar{\Delta}}{s_{\Delta} / \sqrt{5}}$$
- This eliminates the methodological flaw of the earlier one-sample $t$-test against a constant scalar.

---

## 8. Pre-Execution Summary Report

### A. Exact Code Changes Required
1. In `care_capability_v23.py`: Corrected layer normalization indexing from `layer_capabilities[:, layer_idx, :]` to `layer_capabilities[layer_idx, :, :]`.
2. In `care_capability_v23.py`: Implemented dual-architecture forward projection handling both fused 3D tensors (`gate_up_proj` / `down_proj`) and unfused `nn.ModuleList`.
3. In `care_capability_v23.py`: Replaced static `len(moe_blocks[0].experts)` with dynamic `current_num_experts = engine.current_num_experts`.
4. In `run_matched_care_64e.py`: Configured 5-seed paired execution matching Random-$K=5$.

### B. Tests Passed and Failed
- Ran test suite: `venv/bin/python audit_repair_pass/full_64e_validation/tests/test_candidate_selection_64e.py`
- **Result:** **6 passed, 0 failed** (Ran 6 tests in 4.230s).
  - `test_invariant_1_full_64e_shape`: **PASSED**
  - `test_invariant_2_layer_normalization_axis_correctness`: **PASSED**
  - `test_invariant_3_candidate_pool_spans_full_64e`: **PASSED**
  - `test_invariant_4_dynamic_tracking_across_sequential_merges`: **PASSED**
  - `test_invariant_5_zero_stale_indices`: **PASSED**
  - `test_invariant_6_unfused_modulelist_compatibility`: **PASSED**

### C. Remaining Operational Risks
1. **Remote GPU Access:** Direct SSH connectivity to `ubuntu@45.194.3.117` returned permission denied during automated check. Execution must be run via the authenticated user terminal or authorized key on the L40S instance.
2. **GPU Memory Ceiling:** Recomputing capability vectors on 128 sequences across 64 experts incurs transient activations. Slices are processed sequentially in batches of 1 to guarantee VRAM $< 28\text{ GB}$ (well within the 48 GB L40S envelope).

### D. Estimated Runtime
- Profiling 64 experts across 16 layers: ~40s per step.
- Evaluating $K=5$ candidates on 128 calibration chunks: ~3s per step.
- Checkpoint PPL evaluation on 65,536 tokens: ~25s per checkpoint.
- **Estimated Wall Time per Seed (16 steps + 4 PPL checks):** $\approx 710\text{ seconds}$ (~11.8 minutes).
- **Total Runtime for all 5 Seeds:** $\approx 59\text{ minutes}$ on an NVIDIA L40S GPU.

### E. Final Frozen Command and Configuration
```bash
# Executed in the repository root with PYTHONPATH set
python3 audit_repair_pass/full_64e_validation/run_matched_care_64e.py \
    --seeds 101 202 303 404 505 \
    --output_dir audit_repair_pass/full_64e_validation/care_adaptive_64e
```
- **Configuration File:** `benchmark/OLMoE-1B-7B/benchmark_config.yaml`
- **Model:** `allenai/OLMoE-1B-7B-0924` (`bfloat16`)
- **Checkpoints:** `[64, 60, 56, 48]`
- **Budget:** $K = 5$ candidate evaluations per step ($80$ evaluations total per seed)

---

**AUDIT CONCLUSION:**  
Pre-execution validation is complete and verified. The corrected code is fully versioned, all 6 unit and invariant tests have passed, and experimental controls are 100% matched to Random-$K=5$. Execution is halted awaiting user confirmation.
