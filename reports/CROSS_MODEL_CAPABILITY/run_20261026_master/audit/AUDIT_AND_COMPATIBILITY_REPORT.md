# CARE-COM Research Campaign Audit & Architectural Compatibility Report

**Campaign ID:** `run_20261026_master`  
**Host:** `ubuntu@45.194.3.117` (NVIDIA L40S, 48 GB VRAM, Driver 580.95.05, CUDA 13.0)  
**Date:** 2026-10-26  
**Auditor:** Lead ML Systems Engineer & Scientific Benchmark Owner  

---

## 1. Model Registry & Checkpoint Verification

| Model Family | Checkpoint Identifier | Architecture | Layers | Experts/Layer | Routing Top-K | Total Params | Active Params | Dtype | VRAM Footprint | Compatibility Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OLMoE** | `allenai/OLMoE-1B-7B-0924` | `OlmoeForCausalLM` | 16 | 64 | 8 | 6.92B | 1.3B | BF16 | ~14.8 GB | ✅ FULLY COMPATIBLE |
| **JetMoE** | `jetmoe/jetmoe-8b` | `JetMoEForCausalLM` | 24 | 8 | 2 | 8.0B | 2.2B | BF16 | ~17.5 GB | ✅ FULLY COMPATIBLE (Batched Tensors) |
| **Phi-3.5-MoE** | `microsoft/Phi-3.5-MoE-instruct` | `PhiMoEForCausalLM` | 32 | 16 | 2 | 41.9B | 6.6B | 4-bit NF4 | ~22.8 GB | ✅ FULLY COMPATIBLE (Quantized Base) |
| **Qwen3-MoE** | `Qwen/Qwen3-30B-A3B` | `Qwen3MoeForCausalLM` | 48 | 128 | 8 | 30.5B | 3.3B | BF16 | ~59.0 GB | ⚠️ CONDITIONAL (Regime A <=80E pure GPU; >80E streaming) |

---

## 2. Seven Compression Methods & Implementation Fidelity

1. **`care_adaptive` (FAITHFUL):** Dynamic capability recomputation at each merge step + top-5 candidate pool + transactional marginal KL damage evaluation + sequential commit.
2. **`care_static` (FAITHFUL):** Single frozen initial capability ranking + top-5 candidate pool + transactional marginal KL damage evaluation.
3. **`random` (CONTROL):** Uniform random K=5 candidate nomination + transactional marginal KL damage evaluation (matched candidate budget).
4. **`reap_prune` (FAITHFUL PRUNING):** Router-weighted activation saliency accumulation and physical slice deletion. Drops experts without parameter merging.
5. **`sub_moe` (ADAPTED PROXY):** Characteristic activation clustering with router/joint-component reconstruction. Uses CA proxy clustering in offline mode.
6. **`hc_smoe` (PARTIAL PROXY):** Lance-Williams hierarchical agglomerative clustering. Uses router weight cosine or weight-proxy distance rather than token-level activation profiles to prevent memory exhaustion.
7. **`mc_smoe` (PARTIAL):** Hungarian bipartite neuron alignment and core expert selection (Stages 1–4). Stage 5 (low-rank SVD residual pruning) is omitted.

---

## 3. Historical Artifact Provenance & Reuse Accounting

- **OLMoE Language Modeling Sweep:**
  - 63 runs completed across 7 methods $\times$ 3 targets (60E, 56E, 48E) $\times$ 3 seeds (42, 43, 44).
  - Uncompressed baseline WikiText-2 PPL: **10.2804** (65,536 tokens across 128 chunks of 512).
  - Checkpoint integrity: 63/63 runs passed strict serialization and reload checks with 0 missing keys, 0 mismatched keys, exact 0.000000 logit difference.
  - Verification: SHA-256 hashes verified in `reports/BENCHMARK_L40S/run_20261023T105248Z_c4da7cdb/` and `run_20261025T190236Z_fd94e9fb/`.
  - Reused for Track A language modeling; missing reasoning and capability benchmarks to be executed.
- **JetMoE Baseline Results:**
  - Base PPL: **7.1258**.
  - Historical targets: 7E, 6E, 4E. Verified in Exp 15/16.
- **Phi-3.5-MoE Baseline Results:**
  - Base PPL: **4.5700** (4-bit NF4).
  - Historical targets: 14E, 12E, 10E, 8E. Verified in Exp 15/16.
- **Qwen3-30B-A3B Results:**
  - Base PPL: **9.5715**.
  - Verified compression targets: 112E (11.19), 96E (14.81), 80E (19.93), 64E (31.35), 60E (33.51).
  - Retraction: Historical reasoning claims formally retracted due to the 512-token `<think>` truncation bug. Re-evaluating with unconstrained reasoning budget.

---

## 4. Capability-Specific Probing Protocol (Section 18)

- **Strict Three-Role Separation:**
  1. *Capability-Probe Data:* Derived exclusively from `train` / `dev` splits (GSM8K train, MBPP train, MMLU dev, WikiText-2 train).
  2. *Method Calibration Data:* Disjoint validation chunks for micro-eval and activation statistics.
  3. *Held-Out Evaluation Data:* Official `test` splits (GSM8K test, MATH-500 test, MMLU test, MBPP test, WikiText-2 test). Zero overlap guaranteed by dataset hashes.
- **Probe Strategies:**
  - Strategy 1: Generic 10-axis probe (Reference).
  - Strategy 2: Single-capability probes (Math-only, Coding-only, Knowledge-only, LM-only).
  - Strategy 3: Multi-capability combined probe:
    $$C_i^{(\text{multi})} = \frac{1}{\sqrt{K}} [\tilde{C}_i^{(\text{math})}, \tilde{C}_i^{(\text{coding})}, \tilde{C}_i^{(\text{knowledge})}, \tilde{C}_i^{(\text{lm})}]$$
    guaranteeing equal weighting and scale invariance.

---

## 5. Reasoning Generation Protocol (Preventing Previous Flaws)

1. **Reasoning Ceiling:** Minimum 1536 tokens (`max_new_tokens=1536`) to prevent cutting off reasoning traces.
2. **Explicit Answer Delimiters:** Problems formatted with few-shot exemplars ending in `#### <ans>` or `\boxed{<ans>}`.
3. **Strict Answer Scoring:** Score only the closed answer; never use fallback regex grabbing arbitrary trailing digits from incomplete generations.
4. **Diagnostic Tracking:** Explicitly record prompt tokens, generated tokens, stop reasons, truncation count, and parser failure count.
