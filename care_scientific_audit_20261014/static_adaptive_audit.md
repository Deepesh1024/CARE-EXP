# Task 2 Audit Report: Static vs. Adaptive Comparison and Accounting Audit

**Audit Date:** October 14, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifacts Analyzed:**  
- Implementation: [`benchmark/OLMoE-1B-7B/methods/care_com.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/methods/care_com.py)  
- Trajectories: [`benchmark_results/care_static/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_static/trajectory.json) & [`benchmark_results/care_adaptive/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_adaptive/trajectory.json)  
- Aggregation Script: [`benchmark/OLMoE-1B-7B/aggregate_results.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/aggregate_results.py)  
- Published Tables: [`paper_assests/tables/table2_static_vs_adaptive.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/paper_assests/tables/table2_static_vs_adaptive.md) & [`paper_assests/tables/table4_efficiency.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/paper_assests/tables/table4_efficiency.md)  
- Audit Callouts: [`audit/static_vs_adaptive_validation.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/static_vs_adaptive_validation.md) & [`audit/controlled_first_step_validation.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/controlled_first_step_validation.md)

---

## 1. Executive Summary: Major Audit Finding

The comparison between **CARE-Static** and **CARE-Adaptive** contains:
1. **A Valid, Verified Quality Advantage:** Adaptively recomputing capability geometry $C_t$ after each physical merge genuinely improves model quality, reducing Cumulative KL by **15.9%** ($17.69 \to 14.88$) and improving test Perplexity by **0.90 PPL points** ($21.03 \to 20.14$) at the 48-expert target (64 $\to$ 48 experts).
2. **A Fabricated / Corrupted Efficiency Claim:** The paper's claim in Table 4 and Figure 6—that CARE-Adaptive requires **30.4% fewer exact evaluations** (80 vs. 115) and achieves an **80.3% wall-clock time reduction** (~666s vs. ~3387s) due to "dynamically updating topology bounding evaluation depth"—is **factually false**. 
   - Both algorithms evaluated **exactly 5 candidates per step across 16 steps = exactly 80 evaluations**.
   - The reported "115 evaluations" and "3387 seconds" for CARE-Static were caused by **accidental concatenation of an unhandled aborted 7-step test run into `care_static/trajectory.json`**.
   - In truth, CARE-Adaptive took **666.35 seconds**, while the true 16-step CARE-Static run took **301.63 seconds**. Adaptive recomputation was actually **2.21× slower** in wall-clock time due to the computational overhead of updating capability vectors at every step.

---

## 2. Experimental Controls Verification

| Dimension | Verification Status | Forensic Evidence |
|---|---|---|
| **Starting Checkpoint** | **VERIFIED** | Both methods initialize from identical `allenai/OLMoE-1B-7B-0924` weights at $N=64$ (`base_ppl = 10.2878`). |
| **Calibration Set** | **VERIFIED** | Both sample the identical 128 tokens from `results/exp6c/token_vectors/EXP6C_TOKEN_CAPABILITY_VECTORS.parquet` (`random_state=42`). |
| **Evaluation Set** | **VERIFIED** | Both evaluate perplexity on identical WikiText-2 chunks with `seq_len=512, batch_size=8`. |
| **Physical Merge Operator** | **VERIFIED** | Both use identical `PhysicalMergeEngine.merge_experts(i, j)` averaging MLP projections `gate_proj`, `up_proj`, `down_proj`, and router weights. |
| **Target Checkpoints** | **VERIFIED** | Both step sequentially from 64 $\to$ 63 $\to \dots \to$ 48 experts (16 merges total). |
| **Step 1 Candidate Pools** | **VERIFIED** | Both generate the identical candidate set: `[[0, 15], [8, 15], [8, 13], [13, 15], [12, 15]]`. |
| **Step 1 Merge Selection** | **VERIFIED** | Both select the identical first merge: `[8, 13]` with marginal KL damage **0.401733** and capability distance **0.019584**. |

---

## 3. The 115 vs. 80 Evaluation and 3387s vs. 666s Discrepancy

### A. Code Analysis of the Aggregator Bug
In [`benchmark/OLMoE-1B-7B/core/logging.py:L11-17`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/core/logging.py#L11-L17):
```python
if not os.path.exists(log_path):
  with open(log_path, "w") as f:
    json.dump({"config": config, "ppl": {}, "trace": []}, f, indent=2)
```
When `log_path` already exists, `BenchmarkLogger` does **not** clear existing entries; instead, `log_step` executes:
```python
with open(self.log_path, "r") as f:
  data = json.load(f)
data["trace"].append(step_data)
with open(self.log_path, "w") as f:
  json.dump(data, f, indent=2)
```

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

### B. Forensic Anatomy of `care_static/trajectory.json`
Inspection of the raw JSON array in `benchmark_results/care_static/trajectory.json` reveals **23 total step entries**:
- **Run 1 (Aborted Initial Run, Entries 0–6):**
  - Steps 1 through 7 (64 $\to$ 57 experts).
  - Evaluated candidates: $7 \times 5 = \mathbf{35\text{ evaluations}}$.
  - Step 1 took 452.74s, Step 2 took 448.21s, etc. (pre-caching unoptimized run).
  - Total elapsed time logged: **3,085.69 seconds**.
- **Run 2 (Complete 16-Step Production Run, Entries 7–22):**
  - Steps 1 through 16 (64 $\to$ 48 experts).
  - Evaluated candidates: $16 \times 5 = \mathbf{80\text{ evaluations}}$.
  - Total elapsed time logged: **301.63 seconds** (average ~18.8s per step).

### C. True Accounting vs. Paper Claims
Because `aggregate_results.py` iterated unconditionally over all 23 elements where `experts_after >= 48`:
- **Logged Evaluations:** $35 + 80 = \mathbf{115}$.
- **Logged Wall Time:** $3085.69\text{s} + 301.63\text{s} = \mathbf{3,387.33\text{ seconds}}$.

| Metric at 48 Experts | Paper Claim (Table 4 / Fig 6) | Actual Source Record | Corrected Reality |
|---|---|---|---|
| **CARE-Static Evaluations** | 115 exact evaluations | 23 steps in file ($35 + 80$) | **80 exact evaluations** |
| **CARE-Adaptive Evaluations** | 80 exact evaluations | 16 steps in file | **80 exact evaluations** |
| **Evaluation Difference** | -30.4% (claimed advantage) | Aggregator bug artifact | **0.0% (Identical evaluation budget)** |
| **CARE-Static Wall-Clock** | ~3,387 s | $3085.69\text{s} + 301.63\text{s}$ | **301.63 s** |
| **CARE-Adaptive Wall-Clock** | ~666 s | 16 steps in file | **666.35 s** |
| **Time Comparison** | -80.3% (Adaptive faster) | Corrupted by aborted run | **+120.9% (Adaptive is 2.2× SLOWER)** |

> [!CAUTION]
> **Claim Invalidation Notice:** Table 4 and Figure 6 must be retracted or completely rewritten. The scientific claim that *"adaptive topological pruning naturally bounds the required evaluation depth at each step"* is a fictional post-hoc justification constructed to explain a software logging artifact. Both algorithms ran with a static pool size of $K=5$ at every step.

---

## 4. First-Step Discrepancy Reconciliation

In `audit/controlled_first_step_validation.md`, the author reported:
- *Static candidate:* `[8, 13]` | Capability distance ≈ 0.0156 | KL ≈ 0.074
- *Adaptive candidate:* `[13, 15]` | Capability distance ≈ 0.0157 | KL ≈ 0.046

### Forensic Trace of First-Step Selections:
1. **In the actual benchmark run (`care_com.py`):**
   - Both Static and Adaptive evaluated the exact same 5 pairs on Step 1:
     1. `[0, 15]`
     2. `[8, 15]`
     3. `[8, 13]`
     4. `[13, 15]`
     5. `[12, 15]`
   - Both algorithms evaluated all 5 pairs and found `[8, 13]` had the lowest marginal KL ($0.401733$).
   - **Both algorithms selected `[8, 13]` as their first physical merge.**
2. **Where did the `[13, 15]` vs `[8, 13]` narrative originate?**
   - In Step 1, Candidate `[8, 13]` and Candidate `[13, 15]` were both present in the 5-candidate pool.
   - In a preliminary exploratory run (documented in `audit/controlled_first_step_validation.md`), the author examined `[8, 13]` and `[13, 15]` to demonstrate that geometrically similar candidates have different KL damages.
   - However, describing `[8, 13]` as the "Static candidate" and `[13, 15]` as the "Adaptive candidate" was erroneous: both methods generate identical candidate sets at Step 1 because no merges have yet occurred to differentiate their capability states.

---

## 5. Verified Substantive Comparison (Table 2)

While the efficiency narrative in Table 4 collapses, the core algorithmic ablation in Table 2 is **fully authentic and reproducible**:

### Verified Static vs. Adaptive Quality Comparison
| Target Experts | Static PPL | Adaptive PPL | PPL Improvement | Static Cumulative KL | Adaptive Cumulative KL | KL Reduction |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **60** | 12.112 | **11.548** | -0.564 (-4.7%) | 2.477 | **2.036** | **-17.8%** |
| **56** | 14.820 | **13.812** | -1.008 (-6.8%) | 6.820 | **5.143** | **-24.6%** |
| **48** | 21.032 | **20.136** | -0.896 (-4.3%) | 17.694 | **14.881** | **-15.9%** |

### Verified Runtime Accounting
- **CARE-Static Total Compute (64 $\to$ 48):** $301.6\text{s}$ ($\approx 18.8\text{s}$ per merge).
- **CARE-Adaptive Total Compute (64 $\to$ 48):** $666.4\text{s}$ ($\approx 41.6\text{s}$ per merge).
- **Overhead of Recomputation:** $+22.8\text{s}$ per step ($+121\%$ wall time) to recompute token probe representations on the active model.

### Scientific Conclusion for Paper Revision
The paper should transparently frame the true trade-off:
> *"Adaptive recomputation introduces a modest 2.2× computational overhead during offline compression (running at 41.6s vs. 18.8s per step on an RTX 4090), but this one-time investment yields consistent, permanent reductions in cumulative functional damage (16–25% lower KL) and superior test perplexity across all compression targets."*
