# CARE-COM: Toward More Capability-Aware MoE Expert Consolidation
*Investigating capability-geometry signals for more informed expert consolidation under aggressive compression.*

---

## 1. The Research Problem & Approach

Mixture-of-Experts (MoE) architectures achieve superior training efficiency, but their massive aggregate parameter footprints create severe memory bandwidth bottlenecks and deployment costs during high-concurrency inference. Existing MoE merging methods decide which experts to combine using static heuristics or unguided router clustering, risking destructive parameter interference. 

**CARE-COM** investigates whether **functional signals** from expert representations can guide expert consolidation and ultimately help identify risky merges without retraining.

> **Scope:** OLMoE-1B-7B; WikiText-2 perplexity evaluation; MMLU results exploratory (N=100).

---

## 2. Key Empirical Findings (Multi-Seed Benchmark @ 48 Experts)

```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│        19.88 PPL        │        -2.45 PPL        │      6.9–15.0 PPL       │
│      BEST OBSERVED      │        MARGIN VS        │     IMPROVEMENT VS      │
│   MERGING PPL @ 48E     │    RANDOM-K5 CONTROL    │   CLUSTERING PROXIES    │
│  (CARE-Adaptive, n=3)   │  (Directional, n=3)     │ (MC 26.7, HC 31, Sub 35)│
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 3. Method Architecture: Capability-Aware Candidate Selection

```
[ MoE Expert Activations ] ──> [ Functional Distance ] ──> [ Top-K Candidate Pool ]
        X ~ D_calib               D(i, j) = ||C_i - C_j||_2         K = 5 Nearest Pairs
                                                                           │
                                                                           ▼
[ Physical Consolidation ] <── [ Transactional Validation Gate ] <─────────┘
  W_new = (W_i + W_j) / 2               Greedy argmin ΔKL
   (Router Renormalized)
```

---

## 4. Multi-Seed Compression Benchmark (OLMoE-1B-7B @ 48 Retained Experts)

Evaluated across Seeds 42, 43, 44 on WikiText-2 (65,536 tokens, 128 chunks $\times$ 512 tokens, exact token-weighted causal cross-entropy).

| Method | Compression Paradigm | 48E Test PPL ($\mu \pm \sigma$) | Held-out Test KL ($\mu \pm \sigma$) | $\Delta$ PPL vs Base (10.28) | Status / Implementation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Uncompressed Base** | Reference (64E) | **10.2804** | 0.0000 | Baseline | `allenai/OLMoE-1B-7B-0924` |
| **CARE-Adaptive** | Dynamic Geometry | **19.88 ± 0.923** | **0.658 ± 0.038** | +9.60 PPL | Faithful capability re-projection |
| **Random-K5 Control** | Unbiased Merge | **22.33 ± 0.916** | **0.823 ± 0.043** | +12.05 PPL | Matched $K=5$ candidate pool |
| **CARE-Static** | Frozen Geometry | **24.33 ± 0.871** | **0.832 ± 0.051** | +14.05 PPL | Base geometry (no recomputation) |
| **MC-SMoE Proxy** | Hungarian Matching | **26.74 ± 0.000** | **0.955 ± 0.000** | +16.46 PPL | Stages 1–4 neuron alignment (offline) |
| **HC-SMoE Proxy** | Hierarchical Clustering | **31.00 ± 0.000** | **1.082 ± 0.000** | +20.72 PPL | Lance-Williams agglomerative clustering |
| **Sub-MoE Proxy** | Joint SVD Co-activation| **34.84 ± 0.000** | **1.174 ± 0.000** | +24.56 PPL | Router affinity clustering (offline) |

*\*All external baseline proxies implement zero-shot offline stages without post-merge fine-tuning.*

---

## 5. The Three Scientific Pillars

1. **The Problem: Merging Without Damage Estimation**  
   Existing methods merge experts based on router co-activation affinity or static weight distance, without predicting functional interference. Under aggressive consolidation (48 experts, -25% reduction), heuristic clustering baselines collapse (Sub-MoE degrades to 34.84 PPL; HC-SMoE to 31.00 PPL).
2. **Offline Damage Geometry & Motivation**  
   Offline metric analysis (Exp 4) reveals that the physical merge damage matrix has an intrinsic low-rank structure (q=4 metric MDS captures 75% of damage rank variance). This demonstrates that merge interference is geometrically structured, motivating future research on finding genuinely pre-merge observational approximations that do not require prior merge interventions.
3. **Multi-Seed Compression Results**  
   At 48 retained experts on OLMoE, CARE-Adaptive achieved the best observed multi-seed perplexity among the evaluated merging approaches: **19.88 ± 0.923 PPL** versus **22.33 ± 0.916** for Random-K5 (directional, $n=3$). Without dynamic loss feedback, frozen CARE-Static achieves **24.33 ± 0.871 PPL**.

---

## 6. Scope Bounds & Methodological Safeguards

- **Sample Size Bound:** Evaluation at 48 experts is based on $n=3$ seeds; paired $t$-test yields $p=0.1330$. The advantage over Random-K5 is consistent and directional, but not statistically significant.
- **Capability Floor:** In exploratory downstream testing ($N=100$ MMLU), all evaluated merging methods score near the 25% chance floor; capability retention remains unestablished.
- **Observational Boundary:** While offline analysis proves merge damage has low-dimensional metric structure (q=4), zero-shot prediction of damage directly from forward-pass activations remains an open research problem.

---

## 7. Open Research Question for NVIDIA Technical Discussion

> **"Can genuinely pre-merge functional signals identify destructive expert merges without requiring oracle merge measurements?"**
>
> *I would value feedback on: (1) designing pre-merge capability representations that approximate the low-rank damage space, (2) architectural limits of functional distance across non-uniform routing, and (3) hybrid architectures combining pruning for dormant experts with geometry-guided merging for active clusters.*
