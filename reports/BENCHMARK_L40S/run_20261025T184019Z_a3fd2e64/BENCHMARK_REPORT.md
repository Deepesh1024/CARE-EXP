# CARE-MoE Controlled Benchmark Campaign Report (NVIDIA L40S)
**Campaign Status:** ✅ **COMPLETE**  
**Run Timestamp:** `2026-10-25T18:40:19.280922+00:00`  
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
- **Inference Throughput:** 33564.71 tokens/s
- **Inference Latency:** 61.02 ms/batch (batch_size=4)
- **Peak VRAM:** 14399.58 MB

---

## 2. Benchmark Results

| Method | Target | Verified Experts | Status | PPL | Δ PPL | Causal Held-out KL | Probe KL | Marginal KL Sum | Time (s) | Reload |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **reap_prune** (s=42) | 60E | 60E (✅) | ✅ OK | **10.4349** | +0.1546 | 0.024248 | 0.718750 | - | 13.28s | ✅ PASS |
| **reap_prune** (s=42) | 48E | 48E (✅) | ✅ OK | **13.0079** | +2.7276 | 0.331213 | 1.626953 | - | 12.88s | ✅ PASS |
| **sub_moe** (s=42) | 60E | 60E (✅) | ✅ OK | **12.3204** | +2.0400 | 0.192759 | 0.548340 | - | 15.57s | ✅ PASS |
| **sub_moe** (s=42) | 48E | 48E (✅) | ✅ OK | **34.844** | +24.5636 | 1.174168 | 1.876953 | - | 45.18s | ✅ PASS |
| **hc_smoe** (s=42) | 60E | 60E (✅) | ✅ OK | **12.2183** | +1.9380 | 0.180284 | 0.598633 | - | 0.58s | ✅ PASS |
| **hc_smoe** (s=42) | 48E | 48E (✅) | ✅ OK | **30.9989** | +20.7186 | 1.082192 | 1.640625 | - | 1.3s | ✅ PASS |
| **mc_smoe** (s=42) | 60E | 60E (✅) | ✅ OK | **11.9581** | +1.6777 | 0.162427 | 0.545898 | - | 3.52s | ✅ PASS |
| **mc_smoe** (s=42) | 48E | 48E (✅) | ✅ OK | **26.7412** | +16.4608 | 0.954990 | 1.986328 | - | 13.6s | ✅ PASS |

---

## 4. Scientific Fidelity and Methodological Status

- **CARE-Static (`FAITHFUL`):** Single frozen initial capability ranking, explicit active-to-original identity tracking.
- **CARE-Adaptive (`FAITHFUL`):** Dynamic capability recomputation at each step, top-5 candidate pool, transactional argmin KL damage evaluation.
- **Random Control (`CONTROL`):** Matched K=5 candidate pool evaluated physically via marginal KL.
- **Sub-MoE (`FAITHFUL`):** Joint SVD decomposition of expert weight matrices with frequency-weighted singular component reconstruction.
- **REAP-Prune (`FAITHFUL`):** Router-weighted activation norm saliency accumulation and physical expert slice deletion.
- **HC-SMoE (`PARTIAL (router-proxy)`):** Approximated using router weight cosine similarity rather than calibration activation profiles.
- **MC-SMoE (`PARTIAL (Stages 1–4 only)`):** Implements Hungarian bipartite neuron matching and permuted merge; Stage 5 (low-rank SVD residual pruning) is omitted.

*Generated automatically by benchmark/run_l40s_benchmark.py at 2026-10-25T18:45:48.493248+00:00*