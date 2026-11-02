# CARE-MoE Controlled Benchmark Campaign Report (NVIDIA L40S)
**Campaign Status:** ✅ **COMPLETE**  
**Run Timestamp:** `2026-10-23T10:52:48.634891+00:00`  
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
- **Inference Throughput:** 33585.97 tokens/s
- **Inference Latency:** 60.98 ms/batch (batch_size=4)
- **Peak VRAM:** 14400.08 MB

---

## 2. Benchmark Results

| Method | Target | Verified Experts | Status | PPL | Δ PPL | Causal Held-out KL | Probe KL | Marginal KL Sum | Time (s) | Reload |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **care_static** (s=42) | 60E | 60E (✅) | ✅ OK | **12.3204** | +2.0400 | 0.179550 | 0.818359 | 0.219093 | 277.84s | ✅ PASS |
| **care_static** (s=42) | 56E | 56E (✅) | ✅ OK | **16.0929** | +5.8125 | 0.474560 | 1.441406 | 0.539085 | 514.19s | ✅ PASS |
| **care_static** (s=42) | 48E | 48E (✅) | ✅ OK | **24.8612** | +14.5808 | 0.877691 | 2.156250 | 1.111347 | 979.67s | ✅ PASS |
| **care_adaptive** (s=42) | 60E | 60E (✅) | ✅ OK | **12.0226** | +1.7423 | 0.151908 | 0.905273 | 0.198878 | 358.59s | ✅ PASS |
| **care_adaptive** (s=42) | 56E | 56E (✅) | ✅ OK | **14.5643** | +4.2839 | 0.337573 | 1.400391 | 0.451150 | 686.49s | ✅ PASS |
| **care_adaptive** (s=42) | 48E | 48E (✅) | ✅ OK | **19.9192** | +9.6388 | 0.647750 | 1.437500 | 0.807222 | 1259.99s | ✅ PASS |
| **random** (s=42) | 60E | 60E (✅) | ✅ OK | **12.3384** | +2.0581 | 0.186644 | 0.726562 | 0.206396 | 263.69s | ✅ PASS |
| **random** (s=42) | 56E | 56E (✅) | ✅ OK | **15.3996** | +5.1193 | 0.455969 | 1.298828 | 0.468581 | 486.64s | ✅ PASS |
| **random** (s=42) | 48E | 48E (✅) | ✅ OK | **22.9781** | +12.6977 | 0.872798 | 1.490234 | 1.011865 | 938.85s | ✅ PASS |
| **care_static** (s=43) | 60E | 60E (✅) | ✅ OK | **11.9815** | +1.7012 | 0.149706 | 0.976562 | 0.200491 | 286.16s | ✅ PASS |
| **care_static** (s=43) | 56E | 56E (✅) | ✅ OK | **15.2721** | +4.9917 | 0.375245 | 1.224609 | 0.466617 | 530.07s | ✅ PASS |
| **care_static** (s=43) | 48E | 48E (✅) | ✅ OK | **24.8126** | +14.5322 | 0.841487 | 1.523438 | 1.083314 | 933.44s | ✅ PASS |
| **care_adaptive** (s=43) | 60E | 60E (✅) | ✅ OK | **11.807** | +1.5266 | 0.143102 | 0.887695 | 0.170588 | 344.27s | ✅ PASS |
| **care_adaptive** (s=43) | 56E | 56E (✅) | ✅ OK | **13.33** | +3.0497 | 0.250978 | 1.185547 | 0.346970 | 673.99s | ✅ PASS |
| **care_adaptive** (s=43) | 48E | 48E (✅) | ✅ OK | **18.931** | +8.6507 | 0.626223 | 1.742188 | 0.794430 | 1268.09s | ✅ PASS |
| **random** (s=43) | 60E | 60E (✅) | ✅ OK | **12.6193** | +2.3389 | 0.193982 | 0.889648 | 0.224613 | 246.88s | ✅ PASS |
| **random** (s=43) | 56E | 56E (✅) | ✅ OK | **15.1901** | +4.9098 | 0.388943 | 1.187500 | 0.481566 | 499.81s | ✅ PASS |
| **random** (s=43) | 48E | 48E (✅) | ✅ OK | **22.721** | +12.4406 | 0.800391 | 1.380859 | 1.049883 | 928.42s | ✅ PASS |
| **care_static** (s=44) | 60E | 60E (✅) | ✅ OK | **12.5117** | +2.2313 | 0.194961 | 0.904297 | 0.232889 | 287.63s | ✅ PASS |
| **care_static** (s=44) | 56E | 56E (✅) | ✅ OK | **14.8958** | +4.6154 | 0.365460 | 1.268555 | 0.479950 | 525.68s | ✅ PASS |
| **care_static** (s=44) | 48E | 48E (✅) | ✅ OK | **23.3292** | +13.0489 | 0.777886 | 1.878906 | 1.013792 | 969.39s | ✅ PASS |
| **care_adaptive** (s=44) | 60E | 60E (✅) | ✅ OK | **12.2902** | +2.0099 | 0.175391 | 1.130859 | 0.230301 | 359.72s | ✅ PASS |
| **care_adaptive** (s=44) | 56E | 56E (✅) | ✅ OK | **13.7877** | +3.5073 | 0.301859 | 1.361328 | 0.380327 | 683.86s | ✅ PASS |
| **care_adaptive** (s=44) | 48E | 48E (✅) | ✅ OK | **20.7751** | +10.4948 | 0.699609 | 1.861328 | 0.918132 | 1301.42s | ✅ PASS |
| **random** (s=44) | 60E | 60E (✅) | ✅ OK | **12.1706** | +1.8902 | 0.176370 | 0.699219 | 0.190892 | 263.34s | ✅ PASS |
| **random** (s=44) | 56E | 56E (✅) | ✅ OK | **14.9615** | +4.6811 | 0.408023 | 1.121094 | 0.447144 | 504.58s | ✅ PASS |
| **random** (s=44) | 48E | 48E (✅) | ✅ OK | **21.2792** | +10.9988 | 0.795499 | 1.347656 | 0.952655 | 892.03s | ✅ PASS |

---

## 4. Scientific Fidelity and Methodological Status

- **CARE-Static (`FAITHFUL`):** Single frozen initial capability ranking, explicit active-to-original identity tracking.
- **CARE-Adaptive (`FAITHFUL`):** Dynamic capability recomputation at each step, top-5 candidate pool, transactional argmin KL damage evaluation.
- **Random Control (`CONTROL`):** Matched K=5 candidate pool evaluated physically via marginal KL.
- **Sub-MoE (`FAITHFUL`):** Joint SVD decomposition of expert weight matrices with frequency-weighted singular component reconstruction.
- **REAP-Prune (`FAITHFUL`):** Router-weighted activation norm saliency accumulation and physical expert slice deletion.
- **HC-SMoE (`PARTIAL (router-proxy)`):** Approximated using router weight cosine similarity rather than calibration activation profiles.
- **MC-SMoE (`PARTIAL (Stages 1–4 only)`):** Implements Hungarian bipartite neuron matching and permuted merge; Stage 5 (low-rank SVD residual pruning) is omitted.

*Generated automatically by benchmark/run_l40s_benchmark.py at 2026-10-23T15:55:40.512107+00:00*