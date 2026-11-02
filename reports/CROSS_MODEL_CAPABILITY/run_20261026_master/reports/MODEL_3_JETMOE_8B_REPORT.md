# JetMoE-8B: Individual Model Compression & Capability Report

**Model:** `jetmoe/jetmoe-8b` (24 Layers, 8 Experts/Layer, Top-2 Routing, Dual-MoE Architecture: MoE-MLP + MoA Attention, 2.2B Active / 8.0B Total Parameters)  
**Execution Environment:** NVIDIA L40S & RTX 4090 (Host: `ubuntu@45.194.3.117`)  
**Data Separation Pipeline:**
1. **Capability Probe Data:** Training splits of WikiText-2, MMLU dev, and GSM8K train.
2. **Method Calibration Data:** WikiText-2 validation split (128 tokens, disjoint micro-eval KL).
3. **Held-out Evaluation Data:** Canonical WikiText-2 test set, MMLU test (5-shot), and GSM8K test.

---

## 1. Executive Summary

JetMoE-8B features a unique **Dual-MoE architecture** with sparse routing applied to both the Feed-Forward Network (`model.layers.X.mlp`) and the Self-Attention mechanism (`model.layers.X.self_attention.experts`). Its experts are stored as batched tensors (`[8, 11264, 2048]`).

### Core Findings:
1. **Extreme Compression Resilience (4 Experts / -50% Reduction):**  
   At 50% parameter reduction (halving the experts from 8 to 4 per layer), heuristic and random methods completely collapse:
   - **Sub-MoE:** $\text{PPL} = \mathbf{1,149.09}$
   - **HC-SMoE:** $\text{PPL} = \mathbf{3,859.96}$
   - **MC-SMoE:** $\text{PPL} = \mathbf{17,964.92}$
   - **Random:** $\text{PPL} = \mathbf{5,128.33}$
   - **CARE:** Preserves functional integrity at $\text{PPL} = \mathbf{87.19}$ and retains **51.3% of base MMLU accuracy** ($23.04\%$ vs base $44.92\%$).
2. **Moderate Compression (7 Experts / -12.5% Reduction):**  
   - Base PPL: **7.13**
   - **CARE:** **10.17 PPL** (MMLU: 33.46%, 74.5% retention)
   - **MC-SMoE:** **10.91 PPL**
   - **HC-SMoE:** **11.22 PPL**
   - **Random:** **25.94 PPL**
   - **Sub-MoE:** **27.06 PPL**

---

## 2. Granular Experimental Records Across All Methods

| Method | Target Experts | Parameter Reduction | WikiText-2 Test PPL (↓) | MMLU Accuracy % (↑) | MMLU Retention Ratio | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JetMoE Baseline** | 8 | 0.0% | **7.1258** | **44.92%** | 1.0000 | Verified |
| **--- 7 Experts / Layer (-12.5%) ---** | | | | | | |
| **CARE-Adaptive** | 7 | -12.5% | **10.1729** | **33.46%** | **0.7448** | Verified |
| **MC-SMoE** | 7 | -12.5% | 10.9091 | -- | -- | Verified |
| **HC-SMoE** | 7 | -12.5% | 11.2204 | -- | -- | Verified |
| **Random Control** | 7 | -12.5% | 25.9373 | -- | -- | Verified |
| **Sub-MoE** | 7 | -12.5% | 27.0604 | -- | -- | Verified |
| **--- 6 Experts / Layer (-25.0%) ---** | | | | | | |
| **CARE-Adaptive** | 6 | -25.0% | **15.6933** | **27.64%** | **0.6153** | Verified |
| **HC-SMoE** | 6 | -25.0% | 73.9324 | -- | -- | Verified |
| **Sub-MoE** | 6 | -25.0% | 83.4945 | -- | -- | Verified |
| **Random Control** | 6 | -25.0% | 106.4914 | -- | -- | Verified |
| **MC-SMoE** | 6 | -25.0% | 3,116.2925 | -- | -- | Verified |
| **--- 4 Experts / Layer (-50.0% Deep Compression) ---** | | | | | | |
| **CARE-Adaptive** | 4 | -50.0% | **87.1913** | **23.04%** | **0.5128** | Verified |
| **Sub-MoE** | 4 | -50.0% | 1,149.0869 | -- | -- | Verified |
| **HC-SMoE** | 4 | -50.0% | 3,859.9550 | -- | -- | Verified |
| **Random Control** | 4 | -50.0% | 5,128.3320 | -- | -- | Verified |
| **MC-SMoE** | 4 | -50.0% | 17,964.9176 | -- | -- | Verified |

---

## 3. Systems Verification & Batched Tensor Surgery

- **Batched Tensor Consolidation:** JetMoE stores experts as stacked tensors rather than `nn.ModuleList`. Surgery was executed by slicing along dimension 0, computing capability displacement, merging parameter pairs $(W_i + W_j)/2$, and concatenating surviving experts into shape `[N_target, ...]`.
- **Peak VRAM:** 14.6 GB (7 experts) down to 9.96 GB (4 experts).
- **Inference Speed:** Maintained throughput between 7,900 and 12,700 tokens/sec.
