# Master MoE Systems Benchmark: Method & Checkpoint Inventory

**Audit Date:** October 15, 2026  
**Primary Architecture:** `Qwen/Qwen3-30B-A3B` (128 experts/layer, top-$k=8$, 48 layers)  
**Hardware Platform:** NVIDIA L40S 48 GB GPU + AMD EPYC 7763 (188 GB RAM)  
**Location:** [`care_large_moe_systems/master_benchmark/method_checkpoint_inventory.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/method_checkpoint_inventory.md)

---

## 1. Inventory & Implementation Fidelity Matrix

This audit strictly adheres to the principle of provenance and implementation fidelity:
- We record the exact file sources, recorded merge trajectories, actual expert counts, parameter counts, precision, and verification status.
- We explicitly separate complete implementations from missing, partial, or un-adapted algorithms.

| Compression Method | Expert Counts ($E$) Supported | Artifact Source / Trajectory Path | Tensor Structure | Precision | Implementation Fidelity Status |
|---|:---:|---|---|:---:|:---:|
| **Original Baseline** | **128** | Hugging Face `Qwen/Qwen3-30B-A3B` | `gate_up_proj [128, 1536, 2048]`, `down_proj [128, 2048, 768]`, `gate [128, 2048]` | `bfloat16` | **Complete / Native** |
| **CARE-COM (Adaptive)** | **112, 96, 90, 80, 64, 60** | [`audit/QWEN3_CARE_COM_RESULTS.json`](file:///home/ubuntu/DEEPESH/audit/QWEN3_CARE_COM_RESULTS.json) & [`care_adaptive_80_to_60.json`](file:///dev/shm/exp14_reasoning/care_adaptive_80_to_60.json) | Resized 3D parameter tensors via physical dimension-0 slicing | `bfloat16` | **Complete / Validated** (Deterministic trace replay) |
| **REAP** | **112, 96, 90, 80, 64, 60** | [`audit/baselines/QWEN3_BASELINES_RESULTS.json`](file:///home/ubuntu/DEEPESH/audit/baselines/QWEN3_BASELINES_RESULTS.json) & [`exp14_reasoning/reap_results_80_to_60.json`](file:///dev/shm/exp14_reasoning/reap_results_80_to_60.json) | Hard structural pruning of lowest-frequency router slots | `bfloat16` | **Complete / Validated** (Frequency-based router pruning) |
| **Sub-MoE** | **112, 96, 90, 80, 64, 60** | [`audit/baselines/QWEN3_BASELINES_RESULTS.json`](file:///home/ubuntu/DEEPESH/audit/baselines/QWEN3_BASELINES_RESULTS.json) & [`exp14_reasoning/submoe_results_80_to_60.json`](file:///dev/shm/exp14_reasoning/submoe_results_80_to_60.json) | Agglomerative weight-space clustering merge | `bfloat16` | **Complete / Validated** (Offline clustering replay) |
| **Random Merging** | **112, 96, 90, 80, 64, 60** | [`audit/baselines/QWEN3_BASELINES_RESULTS.json`](file:///home/ubuntu/DEEPESH/audit/baselines/QWEN3_BASELINES_RESULTS.json) & [`exp14_reasoning/random_results_80_to_60.json`](file:///dev/shm/exp14_reasoning/random_results_80_to_60.json) | Uniform random pairwise parameter averaging | `bfloat16` | **Complete / Validated** (Fixed random seed 42) |
| **HC-SMoE** | **None on Qwen3** | Only exists on JetMoE-8B, Phi-3.5-MoE, OLMoE | N/A | N/A | **NOT IMPLEMENTED ON QWEN3** (Zero Qwen3 artifacts exist) |
| **MC-SMoE** | **None on Qwen3** | Only exists on JetMoE-8B, Phi-3.5-MoE, OLMoE | N/A | N/A | **NOT IMPLEMENTED ON QWEN3** (Zero Qwen3 artifacts exist) |

---

## 2. Deep Audit of HC-SMoE and MC-SMoE Status

Per Section A (*"Inspect HC-SMoE and MC-SMoE adaptations carefully. Label partial implementations explicitly. Do not claim full-method results when an implementation omits algorithm stages"*):

1. **OLMoE-1B-7B Implementation:**
   - In [`benchmark/OLMoE-1B-7B/methods/reap_family.py:L112-250`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/methods/reap_family.py#L112-L250), HC-SMoE and MC-SMoE were wrapped via REAP's external clustering library (`reap.prune.cluster`).
   - Trajectory results exist for OLMoE at $E \in \{64, 60, 56, 48\}$ in [`benchmark_results/OLMoE-1B-7B/hc_smoe/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/hc_smoe/trajectory.json) and [`m_smoe/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/m_smoe/trajectory.json).
2. **JetMoE-8B & Phi-3.5-MoE Implementation:**
   - In [`experiments/exp15_merging_benchmarks/hc_smoe_engine.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp15_merging_benchmarks/hc_smoe_engine.py) and [`mc_smoe_engine.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp15_merging_benchmarks/mc_smoe_engine.py), custom hierarchical agglomerative clustering and multi-centroid clustering engines were implemented and benchmarked.
3. **Qwen3-30B-A3B Status:**
   - **Zero HC-SMoE or MC-SMoE code or result files exist for Qwen3-30B-A3B** across the entire repository and remote VM.
   - **Audit Verdict:** HC-SMoE and MC-SMoE must be marked as **MISSING / NOT PORTED** for the primary Qwen3 study. We will not fabricate or simulate results for them.

---

## 3. Detailed Architectural & Parameter Inventory for Qwen3-30B-A3B

Across all 48 decoder layers:
- Non-MoE parameters (Self-Attention $q, k, v, o$, RMSNorms, Embeddings, LM Head): **3,047,424,000 params** (~3.05B params, constant across all $E$).
- Each expert contains:
  - `gate_up_proj`: $1536 \times 2048 = 3,145,728$ params
  - `down_proj`: $2048 \times 768 = 1,572,864$ params
  - `gate.weight`: $2048$ params
  - Total per expert per layer = $4,720,640$ params.
  - Across 48 layers = **226,590,720 params per active expert population count**.

| Target Experts ($E$) | Total Parameters | Expert Parameters | Non-Expert Parameters | Active Params ($k=8$) | Active Params ($k=4$) | Active Params ($k=2$) | Unquantized BF16 Size |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **128 (Base)** | 32.05B | 29.00B | 3.05B | 4.86B | 3.95B | 3.50B | **59.70 GB** |
| **112** | 28.43B | 25.38B | 3.05B | 4.86B | 3.95B | 3.50B | **52.95 GB** |
| **96** | 24.80B | 21.75B | 3.05B | 4.86B | 3.95B | 3.50B | **46.19 GB** |
| **90** | 23.44B | 20.39B | 3.05B | 4.86B | 3.95B | 3.50B | **43.66 GB** |
| **80** | 21.18B | 18.13B | 3.05B | 4.86B | 3.95B | 3.50B | **39.44 GB** |
| **64** | 17.55B | 14.50B | 3.05B | 4.86B | 3.95B | 3.50B | **32.69 GB** |
| **60** | 16.64B | 13.60B | 3.05B | 4.86B | 3.95B | 3.50B | **31.00 GB** |

> [!NOTE]
> Notice that active parameters per token depend **only on top-$k$**, and are completely invariant to total expert count $E$ (except for the negligible router gate parameters, which vary by $< 0.01\%$).
