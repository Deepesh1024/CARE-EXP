# CARE-MoE Controlled Benchmark Campaign Report (NVIDIA L40S)
**Campaign Status:** ✅ **COMPLETE**  
**Run Timestamp:** `2026-10-25T19:02:36.865056+00:00`  
**GPU:** NVIDIA L40S (47.37 GB VRAM)  
**Model:** `allenai/OLMoE-1B-7B-0924` (OLMoE-1B-7B) | **Dtype:** `bfloat16`  
**Evaluation Corpus:** WikiText-2 `test` split (65,536 tokens across 128 chunks of 512)  
**Held-out KL:** Causally aligned (`logits[:, :-1, :]` vs `attention_mask[:, 1:]`) over 16 chunks (8,192 tokens)  
**Router Merge Mode:** `uniform`  
**Git Revision:** `unknown` (Dirty: `True`)  
**Calibration Parquet SHA-256:** `82351fe218308d83...` (2866 rows)  

---

## 1. Reference Baseline (64 Experts)
- **Parameters:** 6,919,161,856
- **Perplexity (PPL):** **10.2804**
- **Inference Throughput:** 33659.89 tokens/s
- **Inference Latency:** 60.84 ms/batch (batch_size=4)
- **Peak VRAM:** 14399.58 MB

---

## 2. Benchmark Results

| Method | Target | Verified Experts | Status | PPL | Δ PPL | Causal Held-out KL | Probe KL | Marginal KL Sum | Time (s) | Reload |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **reap_prune** (s=42) | 60E | 60E (✅) | ✅ OK | **10.4349** | +0.1546 | 0.024248 | 0.718750 | - | 13.28s | ✅ PASS |
| **reap_prune** (s=42) | 56E | 56E (✅) | ✅ OK | **10.9287** | +0.6483 | 0.102984 | 1.176758 | - | 13.07s | ✅ PASS |
| **reap_prune** (s=42) | 48E | 48E (✅) | ✅ OK | **13.0079** | +2.7276 | 0.331213 | 1.626953 | - | 12.88s | ✅ PASS |
| **sub_moe** (s=42) | 60E | 60E (✅) | ✅ OK | **12.3204** | +2.0400 | 0.192759 | 0.548340 | - | 15.57s | ✅ PASS |
| **sub_moe** (s=42) | 56E | 56E (✅) | ✅ OK | **15.4374** | +5.1570 | 0.421233 | 0.957031 | - | 28.52s | ✅ PASS |
| **sub_moe** (s=42) | 48E | 48E (✅) | ✅ OK | **34.844** | +24.5636 | 1.174168 | 1.876953 | - | 45.18s | ✅ PASS |
| **hc_smoe** (s=42) | 60E | 60E (✅) | ✅ OK | **12.2183** | +1.9380 | 0.180284 | 0.598633 | - | 0.58s | ✅ PASS |
| **hc_smoe** (s=42) | 56E | 56E (✅) | ✅ OK | **15.317** | +5.0366 | 0.411448 | 1.000000 | - | 0.81s | ✅ PASS |
| **hc_smoe** (s=42) | 48E | 48E (✅) | ✅ OK | **30.9989** | +20.7186 | 1.082192 | 1.640625 | - | 1.3s | ✅ PASS |
| **mc_smoe** (s=42) | 60E | 60E (✅) | ✅ OK | **11.9581** | +1.6777 | 0.162427 | 0.545898 | - | 3.52s | ✅ PASS |
| **mc_smoe** (s=42) | 56E | 56E (✅) | ✅ OK | **14.5643** | +4.2839 | 0.378669 | 0.937500 | - | 7.1s | ✅ PASS |
| **mc_smoe** (s=42) | 48E | 48E (✅) | ✅ OK | **26.7412** | +16.4608 | 0.954990 | 1.986328 | - | 13.6s | ✅ PASS |
| **reap_prune** (s=43) | 60E | 60E (✅) | ✅ OK | **10.4349** | +0.1546 | 0.024248 | 0.770508 | - | 13.16s | ✅ PASS |
| **reap_prune** (s=43) | 56E | 56E (✅) | ✅ OK | **10.9287** | +0.6483 | 0.102984 | 1.119141 | - | 13.04s | ✅ PASS |
| **reap_prune** (s=43) | 48E | 48E (✅) | ✅ OK | **13.0079** | +2.7276 | 0.331213 | 1.716797 | - | 13.04s | ✅ PASS |
| **sub_moe** (s=43) | 60E | 60E (✅) | ✅ OK | **12.3204** | +2.0400 | 0.192759 | 0.586914 | - | 15.48s | ✅ PASS |
| **sub_moe** (s=43) | 56E | 56E (✅) | ✅ OK | **15.4374** | +5.1570 | 0.421233 | 0.983398 | - | 28.25s | ✅ PASS |
| **sub_moe** (s=43) | 48E | 48E (✅) | ✅ OK | **34.844** | +24.5636 | 1.174168 | 1.765625 | - | 45.17s | ✅ PASS |
| **hc_smoe** (s=43) | 60E | 60E (✅) | ✅ OK | **12.2183** | +1.9380 | 0.180284 | 0.607422 | - | 0.58s | ✅ PASS |
| **hc_smoe** (s=43) | 56E | 56E (✅) | ✅ OK | **15.317** | +5.0366 | 0.411448 | 0.925781 | - | 0.82s | ✅ PASS |
| **hc_smoe** (s=43) | 48E | 48E (✅) | ✅ OK | **30.9989** | +20.7186 | 1.082192 | 1.537109 | - | 1.28s | ✅ PASS |
| **mc_smoe** (s=43) | 60E | 60E (✅) | ✅ OK | **11.9581** | +1.6777 | 0.162427 | 0.588867 | - | 3.59s | ✅ PASS |
| **mc_smoe** (s=43) | 56E | 56E (✅) | ✅ OK | **14.5643** | +4.2839 | 0.378669 | 0.884766 | - | 6.84s | ✅ PASS |
| **mc_smoe** (s=43) | 48E | 48E (✅) | ✅ OK | **26.7412** | +16.4608 | 0.954990 | 1.853516 | - | 13.66s | ✅ PASS |
| **reap_prune** (s=44) | 60E | 60E (✅) | ✅ OK | **10.4349** | +0.1546 | 0.024248 | 0.759766 | - | 12.83s | ✅ PASS |
| **reap_prune** (s=44) | 56E | 56E (✅) | ✅ OK | **10.9287** | +0.6483 | 0.102984 | 1.138672 | - | 12.16s | ✅ PASS |
| **reap_prune** (s=44) | 48E | 48E (✅) | ✅ OK | **13.0079** | +2.7276 | 0.331213 | 1.763672 | - | 13.17s | ✅ PASS |
| **sub_moe** (s=44) | 60E | 60E (✅) | ✅ OK | **12.3204** | +2.0400 | 0.192759 | 0.638672 | - | 15.8s | ✅ PASS |
| **sub_moe** (s=44) | 56E | 56E (✅) | ✅ OK | **15.4374** | +5.1570 | 0.421233 | 1.030273 | - | 28.51s | ✅ PASS |
| **sub_moe** (s=44) | 48E | 48E (✅) | ✅ OK | **34.844** | +24.5636 | 1.174168 | 1.996094 | - | 45.29s | ✅ PASS |
| **hc_smoe** (s=44) | 60E | 60E (✅) | ✅ OK | **12.2183** | +1.9380 | 0.180284 | 0.666992 | - | 0.56s | ✅ PASS |
| **hc_smoe** (s=44) | 56E | 56E (✅) | ✅ OK | **15.317** | +5.0366 | 0.411448 | 1.003906 | - | 0.81s | ✅ PASS |
| **hc_smoe** (s=44) | 48E | 48E (✅) | ✅ OK | **30.9989** | +20.7186 | 1.082192 | 1.628906 | - | 1.27s | ✅ PASS |
| **mc_smoe** (s=44) | 60E | 60E (✅) | ✅ OK | **11.9581** | +1.6777 | 0.162427 | 0.606445 | - | 3.52s | ✅ PASS |
| **mc_smoe** (s=44) | 56E | 56E (✅) | ✅ OK | **14.5643** | +4.2839 | 0.378669 | 1.046875 | - | 6.98s | ✅ PASS |
| **mc_smoe** (s=44) | 48E | 48E (✅) | ✅ OK | **26.7412** | +16.4608 | 0.954990 | 1.980469 | - | 14.28s | ✅ PASS |

---

## 4. Scientific Fidelity and Methodological Status

- **CARE-Static (`FAITHFUL`):** Single frozen initial capability ranking, explicit active-to-original identity tracking.
- **CARE-Adaptive (`FAITHFUL`):** Dynamic capability recomputation at each step, top-5 candidate pool, transactional argmin KL damage evaluation.
- **Random Control (`CONTROL`):** Matched K=5 candidate pool evaluated physically via marginal KL.
- **Sub-MoE (`FAITHFUL`):** Joint SVD decomposition of expert weight matrices with frequency-weighted singular component reconstruction.
- **REAP-Prune (`FAITHFUL`):** Router-weighted activation norm saliency accumulation and physical expert slice deletion.
- **HC-SMoE (`PARTIAL (router-proxy)`):** Approximated using router weight cosine similarity rather than calibration activation profiles.
- **MC-SMoE (`PARTIAL (Stages 1–4 only)`):** Implements Hungarian bipartite neuron matching and permuted merge; Stage 5 (low-rank SVD residual pruning) is omitted.

*Generated automatically by benchmark/run_l40s_benchmark.py at 2026-10-25T19:22:50.532778+00:00*