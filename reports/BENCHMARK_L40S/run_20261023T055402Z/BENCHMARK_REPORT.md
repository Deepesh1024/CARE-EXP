# CARE-MoE Controlled Benchmark Campaign Report (NVIDIA L40S)
**Run Timestamp:** `2026-10-23T05:54:02.667198+00:00`  
**GPU:** NVIDIA L40S (47.37 GB VRAM)  
**Model:** `allenai/OLMoE-1B-7B-0924` (OLMoE-1B-7B) | **Dtype:** `bfloat16`  
**Evaluation Corpus:** WikiText-2 `test` split (65,536 tokens across 128 chunks of 512)  
**Router Merge Mode:** `uniform`  
**Git Revision:** ``  

---

## 1. Reference Baseline (64 Experts)
- **Parameters:** 6,919,161,856
- **Perplexity (PPL):** **10.2844**
- **Inference Throughput:** 33577.02 tokens/s
- **Inference Latency:** 60.99 ms/batch (batch_size=4)
- **Peak VRAM:** 14906.46 MB

---

## 2. Benchmark Results

### 2.1 Controlled Matched Comparison & Faithful Methods

| Method | Variant / Budget | Target | Verified Experts | PPL | Δ PPL | Functional KL | Compression Time (s) | Throughput (tok/s) | Reload Parity |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **care_adaptive** | `FAITHFUL` | 60E | 60E (✅) | **12.0298** | +1.7454 | 0.905273 | 109.17s | 33953.96 | ✅ PASS |
| **reap_prune** | `FAITHFUL (Cerebras Saliency Pruning)` | 60E | 60E (✅) | **10.6709** | +0.3865 | 0.843750 | 12.8s | 34142.04 | ✅ PASS |

---

## 3. Scientific Fidelity and Methodological Status

- **CARE-Static (`FAITHFUL`):** Single frozen initial capability ranking, explicit active-to-original identity tracking.
- **CARE-Adaptive (`FAITHFUL`):** Dynamic capability recomputation at each step, top-5 candidate pool, transactional argmin KL damage evaluation.
- **Random Control (`CONTROL`):** Matched K=5 candidate pool evaluated physically via marginal KL.
- **Sub-MoE (`FAITHFUL`):** Joint SVD decomposition of expert weight matrices with frequency-weighted singular component reconstruction.
- **REAP-Prune (`FAITHFUL`):** Router-weighted activation norm saliency accumulation and physical expert slice deletion.
- **HC-SMoE (`PARTIAL (router-proxy)`):** Approximated using router weight cosine similarity rather than calibration activation profiles.
- **MC-SMoE (`PARTIAL (Stages 1–4 only)`):** Implements Hungarian bipartite neuron matching and permuted merge; Stage 5 (low-rank SVD residual pruning) is omitted.

---

## 4. Hardware Scaling & Next Steps for NVIDIA H200
- The L40S runs establish single-seed reproducibility and exact logit parity across compression checkpoints.
- When the NVIDIA H200 GPU becomes available tomorrow, launch multi-seed sweeps (seeds 42, 43, 44, 45, 46) across all 6 methods at full 65,536 token evaluation depth.

*Generated automatically by benchmark/run_l40s_benchmark.py at 2026-10-23T05:57:38.072793+00:00*