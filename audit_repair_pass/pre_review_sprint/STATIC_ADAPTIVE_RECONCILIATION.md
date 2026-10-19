# Phase 1: Static-vs-Adaptive Discrepancy Reconciliation Report

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 18, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifact Location:** [`audit_repair_pass/pre_review_sprint/STATIC_ADAPTIVE_RECONCILIATION.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/pre_review_sprint/STATIC_ADAPTIVE_RECONCILIATION.md)  

---

## 1. Executive Summary & Resolution of the Method-to-Result Discrepancy

Earlier audit notes and review queries noted a contradictory labeling hypothesis:
> *"Reported values include: Static (KL=14.881, PPL=20.136, runtime=301.63s, evaluations=80) vs. Adaptive (KL=17.694, PPL=21.032, runtime=666.35s, evaluations=80). The submitted PDF reportedly associates some of these values with the opposite method labels."*

Our forensic bit-level trace across raw trajectory JSON files ([`benchmark_results/care_static/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_static/trajectory.json) and [`benchmark_results/care_adaptive/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_adaptive/trajectory.json)) establishes the definitive factual ground truth:

1. **Quality Metrics (Cumulative KL and WikiText-2 PPL):**
   - **The submitted PDF labels for Cumulative KL and PPL are 100% FACTUALLY AUTHENTIC to the raw experiment execution.**
   - In the raw execution logs, **CARE-Adaptive** achieved:
     - 60 Experts: Cumulative $\text{KL} = 2.036$, $\text{PPL} = 11.548$
     - 56 Experts: Cumulative $\text{KL} = 5.143$, $\text{PPL} = 13.812$
     - 48 Experts: Cumulative $\text{KL} = \mathbf{14.881}$, $\text{PPL} = \mathbf{20.136}$
   - In the raw execution logs, **CARE-Static** achieved:
     - 60 Experts: Cumulative $\text{KL} = 2.477$, $\text{PPL} = 12.112$
     - 56 Experts: Cumulative $\text{KL} = 6.820$, $\text{PPL} = 14.820$
     - 48 Experts: Cumulative $\text{KL} = \mathbf{17.694}$, $\text{PPL} = \mathbf{21.032}$
   - **The rumor that Static achieved $\text{KL} = 14.881$ / $\text{PPL} = 20.136$ and Adaptive achieved $\text{KL} = 17.694$ / $\text{PPL} = 21.032$ is FALSE.** The PDF did not invert the quality metrics. Adaptive recomputation genuinely reduced cumulative functional damage by $15.9\%$ ($17.694 \to 14.881$) and reduced perplexity by $0.90$ points ($21.032 \to 20.136$).

2. **Computational Cost Metrics (Evaluations and Wall-Clock Time):**
   - **The submitted PDF claims in Table 4 and Appendix C (Static: 115 evals / 3387s vs. Adaptive: 80 evals / 666s, claiming a 30.4% evaluation reduction and 80.3% wall-clock speedup for Adaptive) are FATALLY CORRUPTED by an aggregator logging bug.**
   - Both methods evaluated **exactly 5 candidates per step across 16 steps = exactly 80 evaluations**.
   - The reported "115 evaluations" and "3387 seconds" for CARE-Static were produced because [`benchmark_results/care_static/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_static/trajectory.json) contains **23 total JSON entries**: an aborted preliminary 7-step test run (Entries 0–6: 35 evaluations, 3,085.69 seconds) followed by the completed 16-step production run (Entries 7–22: 80 evaluations, 301.63 seconds).
   - In reality, the clean 16-step CARE-Static run completed in **301.63 seconds** (~18.8s/step). CARE-Adaptive took **666.35 seconds** (~41.6s/step).
   - **CARE-Adaptive is actually 2.21× SLOWER than CARE-Static in wall-clock time**, due to the cost of recomputing capability vectors at each step.

---

## 2. Forensic Trace of Raw Experiment Records

### A. Execution Configuration & Controls
- **Model Checkpoint:** `allenai/OLMoE-1B-7B-0924` (bfloat16, 64 experts per layer, top-$k=8$).
- **Base Model PPL:** $10.2878$ on WikiText-2 (128 evaluation chunks $\times$ 512 tokens).
- **Calibration Set:** 128 token capability vectors sampled from `EXP6C_TOKEN_CAPABILITY_VECTORS.parquet` (`random_state=42`).
- **Target Compression Trajectory:** 64 $\to$ 63 $\to$ 62 $\dots \to$ 48 experts (16 sequential merge steps).
- **Candidate Pool Size:** Fixed at $K = 5$ candidate pairs evaluated per step.
- **Physical Merge Operator:** `PhysicalMergeEngine.merge_experts(i, j)` averaging MLP projections (`gate_proj`, `up_proj`, `down_proj`) and router column weights.

---

### B. Forensic Anatomy of `care_static/trajectory.json`
Inspection of [`benchmark_results/care_static/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_static/trajectory.json) reveals a concatenated trace array of 23 entries:

```
Entries 00-06: Aborted Trial (Pre-caching / Unoptimized forward pass)
  [00] step=1 (64->63): 5 cands, time=452.74s, cum_kl=0.431003
  [01] step=2 (63->62): 5 cands, time=448.77s, cum_kl=0.892791
  [02] step=3 (62->61): 5 cands, time=445.60s, cum_kl=1.516921
  [03] step=4 (61->60): 5 cands, time=440.13s, cum_kl=2.295175
  [04] step=5 (60->59): 5 cands, time=437.22s, cum_kl=3.169442
  [05] step=6 (59->58): 5 cands, time=432.33s, cum_kl=4.096867
  [06] step=7 (58->57): 5 cands, time=428.90s, cum_kl=5.285365
  Subtotal Entries 0-6: 7 steps × 5 = 35 evaluations, 3,085.69 seconds. (ABORTED)

Entries 07-22: Complete 16-Step Production Run (Optimized execution)
  [07] step=1  (64->63): pair=[8, 13],  m_kl=0.401733, cum_kl=0.401733, time=20.29s
  [08] step=2  (63->62): pair=[8, 15],  m_kl=0.479340, cum_kl=0.881073, time=20.06s
  [09] step=3  (62->61): pair=[8, 15],  m_kl=0.724060, cum_kl=1.605133, time=19.91s
  [10] step=4  (61->60): pair=[12, 15], m_kl=0.871735, cum_kl=2.476868, time=19.67s  <-- Target 60E (PPL: 12.112)
  [11] step=5  (60->59): pair=[12, 15], m_kl=0.964844, cum_kl=3.441711, time=20.01s
  [12] step=6  (59->58): pair=[12, 15], m_kl=1.011902, cum_kl=4.453613, time=19.39s
  [13] step=7  (58->57): pair=[8, 13],  m_kl=1.157471, cum_kl=5.611084, time=19.20s
  [14] step=8  (57->56): pair=[8, 13],  m_kl=1.208557, cum_kl=6.819641, time=18.94s  <-- Target 56E (PPL: 14.820)
  [15] step=9  (56->55): pair=[12, 15], m_kl=1.223999, cum_kl=8.043640, time=18.75s
  [16] step=10 (55->54): pair=[8, 15],  m_kl=1.276245, cum_kl=9.319885, time=18.52s
  [17] step=11 (54->53): pair=[12, 15], m_kl=1.365234, cum_kl=10.685120, time=18.34s
  [18] step=12 (53->52): pair=[13, 15], m_kl=1.345459, cum_kl=12.030579, time=18.10s
  [19] step=13 (52->51): pair=[8, 15],  m_kl=1.425415, cum_kl=13.455994, time=17.94s
  [20] step=14 (51->50): pair=[12, 15], m_kl=1.407104, cum_kl=14.863098, time=17.70s
  [21] step=15 (50->49): pair=[8, 15],  m_kl=1.435669, cum_kl=16.298767, time=17.52s
  [22] step=16 (49->48): pair=[12, 15], m_kl=1.394775, cum_kl=17.693542, time=17.28s  <-- Target 48E (PPL: 21.032)
  Subtotal Entries 7-22: 16 steps × 5 = 80 evaluations, 301.63 seconds.
```

---

### C. Forensic Anatomy of `care_adaptive/trajectory.json`
[`benchmark_results/care_adaptive/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_adaptive/trajectory.json) contains a clean, uninterrupted 16-step trace array:

```
Entries 00-15: Complete 16-Step Production Run
  [00] step=1  (64->63): pair=[8, 13],  m_kl=0.401733, cum_kl=0.401733, time=45.81s
  [01] step=2  (63->62): pair=[3, 8],   m_kl=0.465057, cum_kl=0.866791, time=45.34s
  [02] step=3  (62->61): pair=[11, 13], m_kl=0.553558, cum_kl=1.420349, time=44.86s
  [03] step=4  (61->60): pair=[3, 11],  m_kl=0.615784, cum_kl=2.036133, time=44.27s  <-- Target 60E (PPL: 11.548)
  [04] step=5  (60->59): pair=[3, 9],   m_kl=0.668304, cum_kl=2.704437, time=43.58s
  [05] step=6  (59->58): pair=[3, 15],  m_kl=0.731995, cum_kl=3.436432, time=43.10s
  [06] step=7  (58->57): pair=[5, 11],  m_kl=0.792908, cum_kl=4.229340, time=42.56s
  [07] step=8  (57->56): pair=[3, 7],   m_kl=0.914001, cum_kl=5.143341, time=41.92s  <-- Target 56E (PPL: 13.812)
  [08] step=9  (56->55): pair=[3, 5],   m_kl=0.946533, cum_kl=6.089874, time=41.34s
  [09] step=10 (55->54): pair=[0, 11],  m_kl=1.048279, cum_kl=7.138153, time=40.90s
  [10] step=11 (54->53): pair=[3, 15],  m_kl=1.117065, cum_kl=8.255219, time=40.24s
  [11] step=12 (53->52): pair=[3, 15],  m_kl=1.083374, cum_kl=9.338593, time=39.66s
  [12] step=13 (52->51): pair=[4, 12],  m_kl=1.281372, cum_kl=10.619965, time=39.18s
  [13] step=14 (51->50): pair=[3, 15],  m_kl=1.284546, cum_kl=11.904510, time=38.43s
  [14] step=15 (50->49): pair=[0, 14],  m_kl=1.445801, cum_kl=13.350311, time=37.97s
  [15] step=16 (49->48): pair=[3, 12],  m_kl=1.530884, cum_kl=14.881195, time=37.19s  <-- Target 48E (PPL: 20.136)
  Total: 16 steps × 5 = 80 evaluations, 666.35 seconds.
```

---

## 3. Mathematical Recomputation of Cumulative KL

For both methods, recomputing $\sum_{t=1}^s \text{marginal\_kl}_t$ across the steps yields an exact numerical match with zero residual ($\Delta = 0.0 \times 10^0$):

| Target Checkpoint | CARE-Static Recomputed $\sum \Delta\text{KL}$ | CARE-Static Stored `cum_kl` | CARE-Adaptive Recomputed $\sum \Delta\text{KL}$ | CARE-Adaptive Stored `cum_kl` | Reduction in Cumulative KL |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **60 Experts (Step 4)** | **2.476868** | 2.476868 | **2.036133** | 2.036133 | **-17.79%** |
| **56 Experts (Step 8)** | **6.819641** | 6.819641 | **5.143341** | 5.143341 | **-24.58%** |
| **48 Experts (Step 16)** | **17.693542** | 17.693542 | **14.881195** | 14.881195 | **-15.90%** |

This confirms that:
- Cumulative KL values were correctly calculated from the marginal step damage.
- The values reported in the paper (Static: 2.477, 6.820, 17.694 vs. Adaptive: 2.036, 5.143, 14.881) are the exact rounded representations of the verified production runs.

---

## 4. Origin of the Aggregator Bug in `aggregate_results.py`

In [`benchmark/OLMoE-1B-7B/aggregate_results.py:L68-76`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/aggregate_results.py#L68-L76):
```python
for step_data in trace:
  if step_data["experts_after"] >= int(experts):
    if step_data.get("cumulative_kl") is not None:
      cum_kl = step_data["cumulative_kl"]
    time_sec += step_data["wall_time_sec"]
    peak_mem = max(peak_mem, step_data["peak_memory_mb"])
    if step_data.get("candidate_pairs"):
      evaluations += len(step_data["candidate_pairs"])
```

Because `aggregate_results.py` iterated unconditionally over all 23 elements in `care_static/trajectory.json` where `experts_after >= 48`:
- Evaluations summed to: $35\text{ (Run 1)} + 80\text{ (Run 2)} = \mathbf{115\text{ evaluations}}$.
- Wall-clock time summed to: $3,085.69\text{s (Run 1)} + 301.63\text{s (Run 2)} = \mathbf{3,387.32\text{ seconds}}$.
- However, `cum_kl` was set via assignment (`cum_kl = step_data["cumulative_kl"]`), so it picked up the final value from the last step (`17.693542`), leaving the quality metric intact while corrupting the runtime and evaluation counts!

---

## 5. Reconciliation Matrix: Paper Claims vs. Empirical Ground Truth

| Metric / Dimension | Submitted Manuscript (Table 4 / App C / Sec 7) | Actual Raw Production Record | Status / Audit Verdict |
|---|---|---|---|
| **Static 60E (KL / PPL)** | 2.477 / 12.112 | 2.476868 / 12.11158 | **CONFIRMED FACT (Exact Match)** |
| **Adaptive 60E (KL / PPL)** | 2.036 / 11.548 | 2.036133 / 11.54832 | **CONFIRMED FACT (Exact Match)** |
| **Static 56E (KL / PPL)** | 6.820 / 14.820 | 6.819641 / 14.81966 | **CONFIRMED FACT (Exact Match)** |
| **Adaptive 56E (KL / PPL)** | 5.143 / 13.812 | 5.143341 / 13.81207 | **CONFIRMED FACT (Exact Match)** |
| **Static 48E (KL / PPL)** | 17.694 / 21.032 | 17.693542 / 21.03206 | **CONFIRMED FACT (Exact Match)** |
| **Adaptive 48E (KL / PPL)** | 14.881 / 20.136 | 14.881195 / 20.13599 | **CONFIRMED FACT (Exact Match)** |
| **Static Evaluations (48E)** | 115 exact evaluations | 80 exact evaluations | **CONTRADICTED (Software logging artifact)** |
| **Adaptive Evaluations (48E)**| 80 exact evaluations | 80 exact evaluations | **CONFIRMED FACT (5 cands × 16 steps)** |
| **Evaluation Advantage** | -30.4% fewer evals for Adaptive | 0.0% difference (Both K=5) | **CONTRADICTED (Fictional claim)** |
| **Static Wall Time (48E)** | ~3,387 seconds | 301.63 seconds | **CONTRADICTED (Contaminated by aborted run)** |
| **Adaptive Wall Time (48E)**| ~666 seconds | 666.35 seconds | **CONFIRMED FACT** |
| **Runtime Comparison** | -80.3% time (Adaptive faster) | +120.9% time (Adaptive 2.2× SLOWER)| **CONTRADICTED (Adaptive is slower)** |

---

## 6. Distinguishing Confirmed Facts, Probable Explanations, and Unresolved Questions

### Confirmed Facts
1. **The method-to-result mapping for quality is authentic:** Adaptive is 14.881 / 20.136; Static is 17.694 / 21.032. The paper did not swap the quality labels.
2. **Recomputing capability vectors improves compression quality:** At all three targets (60E, 56E, 48E), CARE-Adaptive consistently achieves lower cumulative KL and lower test perplexity.
3. **The efficiency claim is an artifact of logging concatenation:** The 115-evaluation / 3387-second count for Static is 100% explained by the presence of 7 unhandled aborted steps (35 evals, 3085.69s) in `care_static/trajectory.json`.
4. **Adaptive recomputation has a real compute cost:** Updating the capability representation $C_t = F(M_{t+1}; \mathcal{P})$ adds ~22.8 seconds per step, making CARE-Adaptive 2.21× slower than CARE-Static in offline compression wall-clock time.

### Probable Explanations
1. **Origin of the Conflicting Method Label Rumor:** When auditors first noticed that Static took 301.63s and had 80 evaluations while Adaptive took 666.35s and had 80 evaluations, an informal summary juxtaposed "Static: 301.63s, 80 evals" with the best quality numbers (14.881, 20.136), leading to the mistaken impression that Static had achieved 14.881 and that the submitted PDF had inverted them. Tracing the raw step objects definitively refutes this: the step objects with KL summing to 14.881 have `"method": "care_adaptive"`, while the step objects summing to 17.694 have `"method": "care_static"`.

### Unresolved Questions & Required Follow-Ups
1. **Multi-Seed Robustness:** Both Static and Adaptive evaluations were executed as a single run ($N=1$) under a fixed calibration batch. Does the 15.9% advantage of Adaptive over Static hold across multiple random calibration draws? (Addressed in Phase 3 Matched Random-K ablation).
