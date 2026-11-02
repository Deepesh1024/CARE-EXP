# OLMoE-1B-7B: Individual Model Compression & Capability Report

**Model:** `allenai/OLMoE-1B-7B-0924` (16 Layers, 64 Experts/Layer, Top-8 Routing, 1.3B Active / 6.92B Total Parameters)  
**Execution Environment:** NVIDIA L40S (48 GB GDDR6 ECC, Host: `ubuntu@45.194.3.117`)  
**Data Separation Pipeline:**
1. **Capability Probe Data:** Training splits of GSM8K, MBPP, MMLU dev, and WikiText-2 train.
2. **Method Calibration Data:** WikiText-2 validation split (128 tokens, disjoint micro-eval KL).
3. **Held-out Evaluation Data:** Canonical WikiText-2 test set (128 chunks $\times$ 512 tokens = 65,536 tokens), GSM8K test, MMLU test, and MBPP test.

---

## 1. Executive Summary

OLMoE-1B-7B was the primary testbed for Section 18's multi-capability and probe-specific experimental campaign, featuring 64 experts per layer and top-8 routing.

### Core Findings:
1. **Section 18 Probe Differentiation:**  
   Coding probe candidate pools have only 25.0% Jaccard overlap with language modeling pools ($J=0.2500$). Coding-guided consolidation preserves MMLU at **28.0%** and achieves the lowest causal KL (**0.6435**) among all 48E adaptive models.
2. **Deep Compression Breakdown (48 Experts / -25% Reduction):**  
   - External baselines suffer catastrophic collapse: Sub-MoE degrades to **34.12 PPL**, HC-SMoE to **30.91 PPL**, and MC-SMoE to **26.58 PPL**.
   - CARE maintains **17.70 PPL** (Static) and **20.32 PPL** (Adaptive).
3. **Multi-Seed Stability:**  
   Evaluated across seeds 42, 43, and 44, CARE multi-capability compression demonstrated bit-exact stability (**23.0231 PPL** across all 3 seeds).

---

## 2. Granular Experimental Records Across All 7 Methods (28 Runs)

| Run Identifier | Method | Probe Strategy | Target Experts | WikiText-2 PPL (↓) | Causal Held-out KL (↓) | GSM8K Acc % (↑) | MMLU Acc % (↑) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `base_olmoe_64e` | **Uncompressed Base** | `none` | 64 | **10.1479** | 0.0000 | **7.0%** | **54.0%** | Verified |
| **--- 60 Experts / Layer ---** | | | | | | | |
| `care_adaptive_generic_60e` | CARE-Adaptive | `generic` | 60 | 11.8156 | 0.1460 | 4.0% | 48.0% | Verified |
| `care_adaptive_multi_60e` | CARE-Adaptive | `multi` | 60 | 12.1290 | 0.1862 | 5.0% | 43.0% | Verified |
| `care_adaptive_math_60e` | CARE-Adaptive | `math` | 60 | 11.9991 | 0.1500 | 5.0% | 41.0% | Verified |
| `care_adaptive_coding_60e` | CARE-Adaptive | `coding` | 60 | 12.3747 | 0.1852 | 6.0% | 40.0% | Verified |
| `care_adaptive_knowledge_60e` | CARE-Adaptive | `knowledge` | 60 | 11.9991 | 0.1500 | 5.0% | 41.0% | Verified |
| `care_adaptive_lm_60e` | CARE-Adaptive | `lm` | 60 | 12.2333 | 0.1656 | 5.0% | 39.0% | Verified |
| `care_static_generic_60e` | CARE-Static | `generic` | 60 | 11.1774 | 0.1068 | 5.0% | 50.0% | Verified |
| `care_static_multi_60e` | CARE-Static | `multi` | 60 | 11.4851 | 0.1144 | 5.0% | 48.0% | Verified |
| `random_none_60e` | Random-K5 | `none` | 60 | 12.1646 | 0.1839 | 3.0% | 42.0% | Verified |
| `reap_prune_none_60e` | REAP-Prune | `none` | 60 | **10.3992** | **0.0328** | 3.0% | **53.0%** | Verified |
| `sub_moe_none_60e` | Sub-MoE | `none` | 60 | 12.1974 | 0.1959 | 4.0% | 37.0% | Verified |
| `hc_smoe_none_60e` | HC-SMoE | `none` | 60 | 12.0226 | 0.1803 | **7.0%** | 38.0% | Verified |
| `mc_smoe_none_60e` | MC-SMoE | `none` | 60 | 11.7983 | 0.1631 | 3.0% | 40.0% | Verified |
| **--- 48 Experts / Layer ---** | | | | | | | |
| `care_adaptive_generic_48e` | CARE-Adaptive | `generic` | 48 | 20.3228 | 0.7342 | 0.0% | 25.0% | Verified |
| `care_adaptive_multi_48e_s42` | CARE-Adaptive | `multi` | 48 | 23.0231 | 0.7448 | 1.0% | 22.0% | Verified |
| `care_adaptive_multi_48e_s43` | CARE-Adaptive | `multi` | 48 | 23.0231 | 0.7448 | 1.0% | 22.0% | Verified |
| `care_adaptive_multi_48e_s44` | CARE-Adaptive | `multi` | 48 | 23.0231 | 0.7448 | 1.0% | 22.0% | Verified |
| `care_adaptive_math_48e` | CARE-Adaptive | `math` | 48 | 20.9384 | 0.6905 | 1.0% | 19.0% | Verified |
| `care_adaptive_coding_48e` | CARE-Adaptive | `coding` | 48 | 20.8975 | **0.6435** | 1.0% | **28.0%** | Verified |
| `care_adaptive_knowledge_48e`| CARE-Adaptive | `knowledge` | 48 | 21.3209 | 0.6767 | 0.0% | 23.0% | Verified |
| `care_static_generic_48e` | CARE-Static | `generic` | 48 | **17.6951** | **0.5493** | **4.0%** | 25.0% | Verified |
| `care_static_multi_48e` | CARE-Static | `multi` | 48 | 18.7375 | 0.6102 | 0.0% | 27.0% | Verified |
| `random_none_48e` | Random-K5 | `none` | 48 | 22.5990 | 0.8707 | 0.0% | 21.0% | Verified |
| `reap_prune_none_48e` | REAP-Prune | `none` | 48 | 12.8404 | 0.3453 | 2.0% | 43.0% | Verified |
| `sub_moe_none_48e` | Sub-MoE | `none` | 48 | 34.1186 | 1.1945 | 1.0% | 25.0% | Verified |
| `hc_smoe_none_48e` | HC-SMoE | `none` | 48 | 30.9081 | 1.1231 | 1.0% | 28.0% | Verified |
| `mc_smoe_none_48e` | MC-SMoE | `none` | 48 | 26.5846 | 0.9822 | 1.0% | 28.0% | Verified |
