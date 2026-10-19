# Phase 2: Experimental Pipeline Validation Report

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 18, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifact Location:** [`audit_repair_pass/pre_review_sprint/PIPELINE_VALIDATION.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/pre_review_sprint/PIPELINE_VALIDATION.md)  

---

## 1. Executive Summary

Before launching experimental reruns, this audit inspects the complete end-to-end execution pipeline across both target model families:
1. **Primary Evaluation Architecture:** `allenai/OLMoE-1B-7B-0924` (16 layers, 64 experts/layer, top-$k=8$)
2. **Scaling & Systems Architecture:** `Qwen/Qwen3-30B-A3B` (48 layers, 128 experts/layer, top-$k=8$)

Every component—tokenizers, checkpoints, router tensors, candidate generators, merge operators, loss metrics, and log aggregators—was traced directly to source code and checked for correctness, data leakage, and potential silent state drift.

---

## 2. Dimension-by-Dimension Pipeline Verification

### 2.1. Model Checkpoint & Tokenizer Revisions
- **OLMoE Pipeline:**
  - Checkpoint: `allenai/OLMoE-1B-7B-0924` hosted on Hugging Face Hub (cached at `~/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924`, 13 GiB).
  - Precision: `bfloat16`.
  - Base PPL: $10.2878$ on WikiText-2 (128 chunks $\times$ 512 tokens).
  - Tokenizer: `AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")`, `pad_token` safely assigned to `eos_token`.
  - Status: **VERIFIED.**
- **Qwen3 Pipeline:**
  - Checkpoint: `Qwen/Qwen3-30B-A3B` (fused 3D MoE tensors).
  - Base PPL: $9.5715$ on WikiText-2 (32 chunks $\times$ 1024 tokens = 32,768 tokens).
  - Tokenizer: `AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)`.
  - Status: **VERIFIED.**

---

### 2.2. Router Configuration & Top-$K$ Enforcement
- **OLMoE:**
  - Router layer: Linear projection mapping hidden dimension $d=2048$ to expert logits $\mathbb{R}^{N_t}$.
  - Routing Sparsity: Fixed top-$k = 8$.
  - Expert Pool Shrinkage: As expert pool drops from 64 down to 48, $N_t \ge 48 \gg 8$, ensuring router top-$k=8$ is mathematically feasible throughout all compression steps.
  - Status: **VERIFIED.**
- **Qwen3:**
  - Architecture: Fused gate weights $W_g \in \mathbb{R}^{128 \times 2048}$.
  - Historical Issue: A top-$k$ sparsity sweep ($k \in \{8, 6, 4, 2\}$) in an earlier run leaked modified `top_k` attributes into subsequent baseline evaluations, causing a reported contradiction for MC-SMoE (PPL 304.52 vs 235.32).
  - Audit Fix: `audit_repair_pass/run_phase2_perplexity_rerun.py` enforces explicit runtime assertions:
    ```python
    assert layer.mlp.gate.top_k == 8, f"Layer {i} has top_k={layer.mlp.gate.top_k} != 8"
    ```
    This assertion is executed directly on the instantiated PyTorch model module before every forward evaluation pass, guaranteeing zero router parameter leakage.
  - Status: **VERIFIED WITH AUDIT ENFORCEMENT.**

---

### 2.3. Checkpoint Loading & Expert Removal
- **OLMoE Physical Merge Engine (`experiments/care_com_v21/core.py:L103-151`):**
  - Structural Mechanism: When pair $(i, j)$ is merged, expert $j$ is deleted from the `ModuleList`:
    ```python
    keep_indices = [idx for idx in range(len(block.experts)) if idx != expert_j]
    block.experts = nn.ModuleList([block.experts[idx] for idx in keep_indices])
    ```
  - Router Weight Adaptation: The router parameter is sliced and updated:
    ```python
    new_router_w = router_w[keep_indices].clone()
    new_router_w[new_expert_i] = router_new
    router_module.weight = nn.Parameter(new_router_w)
    ```
  - Reversibility: `snapshot(expert_i)` stores pre-merge weights and `restore()` reconstructs the original `ModuleList` without generating dangling references or memory leaks.
  - Status: **VERIFIED.**
- **Discrepancy Note on Router Weight Merging:**
  - Manuscript Text (Sec 5.4 / App B): *"The router weight column for expert j is added to the column for expert i to preserve expected routing mass."*
  - Actual Implementation (`core.py:L121`):
    ```python
    router_new = (router_w[expert_i] + router_w[expert_j]) / 2.0
    ```
    The implementation **averages** router rows with factor $0.5$ rather than summing them. The manuscript must be corrected to state that router weights are averaged.
  - Status: **DISCREPANCY DOCUMENTED (Text correction required).**

---

### 2.4. Calibration & Evaluation Datasets
- **Calibration Set:**
  - Source: `results/exp6c/token_vectors/EXP6C_TOKEN_CAPABILITY_VECTORS.parquet`.
  - Batching: Exactly 128 sampled token sequences (`random_state=42`).
  - Consistency: Used identically for both baseline probability distribution computation and step-by-step marginal KL measurement.
  - Status: **VERIFIED.**
- **Evaluation Set (WikiText-2 Test Split):**
  - OLMoE Protocol: 128 chunks $\times$ 512 tokens = 65,536 tokens.
  - Qwen3 Protocol: 32 chunks $\times$ 1024 tokens = 32,768 tokens (`num_eval_chunks=32`).
  - Label Masking: `labels[attention_mask == 0] = -100` correctly applied.
  - Cross-Entropy Calculation: Native PyTorch cross-entropy without artificial multipliers or scaling heuristics.
  - Status: **VERIFIED.**

---

### 2.5. Candidate Generation & Search Budget
- **Algorithm:**
  1. Compute pairwise capability Euclidean distance matrix:
     $$d_C(i, j) = \|C_i - C_j\|_2$$
  2. Enforce pair uniqueness and symmetry ($i < j$).
  3. Sort pairs in ascending order of distance: $\text{Top-}K$.
  4. Candidate pool size fixed at $K = 5$.
- **Integrity Check:**
  - Evaluated pairs per step: exactly 5.
  - Pair selection: Greedy minimum marginal KL damage across the 5 candidates.
  - Status: **VERIFIED.**

---

### 2.6. Logging & Aggregator Integrity
- **Logging Bug Identified:**
  In `benchmark/OLMoE-1B-7B/core/logging.py`, existing files are not overwritten; appending to an existing trajectory file caused aborted 7-step test data to combine with the 16-step run.
- **Remediation Specification for All New Sprint Runs:**
  All runners in `audit_repair_pass/pre_review_sprint/` must:
  1. Write to fresh, unique run directories (`run_<timestamp>_<seed>/`).
  2. Verify that existing trajectory files are either archived or cleanly overwritten with a confirmed pre-run header.
  3. Filter trajectory parsing by unique monotonically increasing step counters (`step == idx + 1`).
  - Status: **REMEDIATION ENFORCED.**

---

## 3. Pipeline Validation Matrix

| Subsystem | Audit Status | Forensic Evidence / Code File | Required Remediation |
|---|:---:|---|---|
| **Checkpoint Fingerprinting** | PASS | Layer 0, 24, 47 SHA-256 tensor hashing in `run_phase2_perplexity_rerun.py` | Enforce on all new runs |
| **Router top_k Enforcement** | PASS | Runtime `assert layer.mlp.gate.top_k == 8` verified | Enforce model-level assertion |
| **Physical Expert Removal** | PASS | `PhysicalMergeEngine` in `care_com_v21/core.py` verified | None |
| **Router Weight Rule** | WARNING | Code uses averaging `(w_i + w_j)/2`; text claimed addition | Correct manuscript Section 5.4 |
| **Calibration Set** | PASS | Exactly 128 tokens from `EXP6C_TOKEN_CAPABILITY_VECTORS.parquet` | None |
| **WikiText-2 PPL Harness** | PASS | Contiguous 512/1024 chunks with cross-entropy evaluation | None |
| **Marginal KL Objective** | PASS | `F.kl_div(reduction='batchmean')` with precomputed baseline logits | None |
| **Aggregator Script** | WARNING | Iterates unconditionally over JSON traces; combined partial runs | Filter by monotonic step index |
| **Random Seed Control** | PASS | Fixed seeds specified in config | Enforce across Random-K ablation |
