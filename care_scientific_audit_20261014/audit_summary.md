# CARE-COM Scientific Audit: Executive Summary & Consolidated Findings

**Audit Date:** October 14, 2026  
**Audit Scope:** CARE-COM Research Project & Submitted ICLR 2027 Manuscript  
**Auditor:** Antigravity Autonomous Audit Agent (DeepMind Advanced Agentic Coding)  
**Execution Constraints:** CPU-Only Static Analysis, Zero GPU Usage, Zero Running Process Interruption, Zero Modification to Submitted/Source Files  
**Audit Directory:** [`care_scientific_audit_20261014/`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/)

---

## 1. Executive Summary

This independent scientific audit examined the codebase, raw experimental logs, trajectories, and submitted paper artifacts of the **CARE-COM** project. The audit was conducted strictly on CPU without executing model inference, training, or modifying any existing repository artifacts.

The objective was to evaluate the empirical veracity and methodological rigor supporting the paper's three central scientific claims:
1. *Capability geometry predicts merge-induced functional damage.*
2. *Geometry-guided candidate generation adds value beyond alternative candidate-selection strategies.*
3. *Recomputing capability geometry after each physical merge improves compression outcomes compared with a frozen initial ranking.*

### Summary Verdict
- **Claim 1 (Geometry Predicts Damage):** **Partially Verified.** The reported Spearman correlations ($\rho_B = 0.7504$, $\rho_C = 0.8146$, pair-level $\rho = 0.7657$) are mathematically exact and replicate to 4 decimal places from source records. However, a critical methodological confound exists: the "Capability Geometry" evaluated in Experiment 4 is derived from Metric MDS on *ground-truth physical Oracle KL interventions* requiring 43 physical merges per test expert, rather than zero-shot observational token probes.
- **Claim 2 (Geometry Adds Value Beyond Selection Alternatives):** **Unverified.** The paper compares CARE ($K=5$ candidate pool + greedy Oracle KL search) against an uncontrolled "Random" baseline that executes $K=1$ random merge with zero search ($0$ Oracle KL evaluations). The isolated contribution of geometric candidate nomination versus random candidate nomination under matched local search remains untested.
- **Claim 3 (Recomputing Capability State Outperforms Static):** **Quality Verified, Efficiency Contradicted.** Adaptive recomputation genuinely improves compression quality over a static frozen ranking under matched conditions (PPL $20.14$ vs $21.03$, Cumulative KL $14.88$ vs $17.69$). However, the efficiency claims in Table 4 and Figure 6 (30.4% fewer evaluations, 80.3% wall-clock reduction) are **factually false**, caused by a software aggregator bug that concatenated an aborted 7-step trial into `care_static/trajectory.json`. In reality, both methods evaluated exactly 80 candidates, and CARE-Adaptive was **2.21× slower** ($666.35\text{s}$ vs $301.63\text{s}$).

---

## 2. Five Most Consequential Findings (Ranked by Scientific Importance)

### Finding 1 (Rank 1 — Invalid Efficiency Claims in Table 4 & Figure 6)
- **Paper Claim:** CARE-Adaptive achieves an 80.3% wall-clock speedup (~666s vs ~3387s) and 30.4% fewer evaluations (80 vs 115) because "adaptive topological pruning naturally bounds the required evaluation depth at each step."
- **Forensic Evidence:** In [`benchmark_results/care_static/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/care_static/trajectory.json), `BenchmarkLogger` appended the 16-step production run to an earlier aborted 7-step test run (35 evals, 3,085.7s), yielding 23 total entries. [`benchmark/OLMoE-1B-7B/aggregate_results.py:L59-67`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/aggregate_results.py#L59-L67) summed all entries where `experts_after >= 48`, reporting $35 + 80 = 115$ evaluations and $3,085.7\text{s} + 301.6\text{s} = 3,387.3\text{s}$.
- **Ground Truth:** Both algorithms evaluated **exactly 5 candidates per step $\times$ 16 steps = 80 evaluations**. The true wall-clock time for CARE-Static was **301.63 seconds**, whereas CARE-Adaptive took **666.35 seconds**. Adaptive capability recomputation was **2.21× slower**, not 5× faster.
- **Impact:** Table 4 and Figure 6 are empirically invalid. The theoretical narrative describing "bounded evaluation depth" was a post-hoc rationalization of a logging bug.

### Finding 2 (Rank 2 — Transductive Oracle Embedding in Experiment 4 / Table 1)
- **Paper Claim:** Capability geometry predicts merge-induced functional damage ($\rho = 0.7504$ to $0.8146$) across unseen, expert-disjoint cross-validation folds.
- **Forensic Evidence:** In [`experiments/experiment4/mds_embedding.py:L62-95`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4/mds_embedding.py#L62-95), out-of-sample test experts are embedded into the 4D MDS space via L-BFGS-B minimization against their Oracle KL distance to all 43 training experts.
- **Ground Truth:** Placing 21 test experts into the geometric space required evaluating $21 \times 43 = \mathbf{903\text{ physical Oracle merge interventions}}$ per fold—over 4× more physical merges than the entire test set being predicted ($\binom{21}{2} = 210$). 
- **Impact:** Table 1 does not demonstrate that zero-shot observational probe geometry predicts merge damage; it demonstrates that the *ground-truth physical intervention matrix itself* has low-rank structure ($q=4$). The paper must clarify this distinction.

### Finding 3 (Rank 3 — Confounded Candidate-Generation Baseline / Missing Random-K Control)
- **Paper Claim:** Geometry-guided candidate nomination adds significant value beyond alternative selection strategies.
- **Forensic Evidence:** In [`benchmark/OLMoE-1B-7B/methods/random_com.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/methods/random_com.py), the "Random" baseline selects 1 random pair per step and performs $K=1$ merge with 0 evaluations. CARE-Adaptive nominates $K=5$ candidate pairs and performs exact Oracle KL evaluation on 128 calibration tokens to pick the minimum.
- **Ground Truth:** The performance difference (PPL 20.14 vs 38.10) conflates candidate pool quality with the optimization power of local greedy search.
- **Impact:** Claim 2 is scientifically unverified until a **Matched Random-$K$ Control** ($K=5$ random candidates + identical Oracle KL selection) is executed.

### Finding 4 (Rank 4 — Cross-Regime Conflation in Figure 3 & Text)
- **Paper Claim:** 186 candidate pairs with nearly identical capability distance ($|\Delta d| < 10^{-4}$) exhibit Oracle KL damages spanning 0.001522 to 0.007096 ($4.66\times$), illustrated by Candidates `[8, 13]` ($d=0.0156, \text{KL}=0.074$) and `[13, 15]` ($d=0.0157, \text{KL}=0.046$).
- **Forensic Evidence:** In [`audit/experiment4_pair_level.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/experiment4_pair_level.csv), the 186-pair subset around median $d_C = 0.002214$ is verified to 6 decimal places ($4.6596\times$ ratio). However, Candidates `[8, 13]` and `[13, 15]` have distances $\sim 0.0156$ and KL values $\sim 0.046–0.074$—an order of magnitude outside the 186-pair distribution.
- **Ground Truth:** The illustrative pairs were pulled from an online sequential compression run on full forward passes, whereas the 186-pair distribution came from offline middle-layer MDS.
- **Impact:** An editorial cross-experimental conflation that creates the appearance of contradictory data. Reconciled by replacing the narrative examples with verified internal pairs: `[18, 37]` ($d=0.002157, \text{KL}=0.007096$) vs `[8, 57]` ($d=0.002159, \text{KL}=0.001676$), achieving a **4.23× damage ratio** with $|\Delta d| < 3 \times 10^{-6}$.

### Finding 5 (Rank 5 — Statistical Power & Dyadic Clustering in Experiment 4)
- **Paper Claim:** Asymptotic significance ($p < 10^{-300}$) for geometry prediction across 3,255 test pairs.
- **Forensic Evidence:** In [`results/exp4/final_report.md:L164-169`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/final_report.md#L164-L169), the authors noted that the 5 partitions are the independent statistical units, and folds within partitions share correlation.
- **Ground Truth:** Pairs sharing an expert (e.g., $(e_1, e_2)$ and $(e_1, e_3)$) exhibit dyadic non-independence. Treating 3,255 pairs as independent i.i.d. draws artificially deflates standard errors. At the true unit of independence ($N=5$ partitions), the Wilcoxon signed-rank test yields $p = 0.0312$, which is significant at $\alpha = 0.05$ but offers limited statistical power.

---

## 3. Claim-to-Evidence Status Matrix

| Major Paper Claim | Source Location | Audit Verdict | Key Scientific Limitation | Safe Formulated Wording |
|---|---|:---:|---|---|
| **Claim 1: Geometry Predicts Damage** | [`results/exp4/partition_metrics.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/partition_metrics.csv) | **Partially Verified** | MDS embedding requires 43 Oracle merge evaluations per test expert; dyadic pair dependence. | Low-rank geometric embeddings derived from observed merge interventions correlate with unobserved pair damage (mean $\rho=0.7504$ across 5 expert-disjoint partitions). |
| **Claim 2: Geometry Outperforms Alternatives** | [`benchmark_results/OLMoE-1B-7B/summary/results.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/summary/results.csv) | **Unverified** | Confounded by $K=1$ (zero search) vs $K=5$ (greedy Oracle KL search). | CARE-Adaptive substantially outperforms unguided random merging without search (PPL 20.14 vs 38.10), but comparison against matched local search is pending. |
| **Claim 3a: Adaptive Recomputation Quality** | [`benchmark_results/OLMoE-1B-7B/summary/results.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/summary/results.csv) | **Verified** | $N=1$ trajectory from deterministic probe sampling. | Dynamically recomputing capability state after each merge reduces cumulative KL by 15.9% and improves perplexity by 0.90 PPL points over frozen static ranking. |
| **Claim 3b: Adaptive Search Efficiency (Table 4 / Fig 6)** | [`paper_assests/tables/table4_efficiency.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/paper_assests/tables/table4_efficiency.md) | **Contradicted** | Software aggregator bug concatenated aborted 7-step trial; both evaluate 80 candidates; Adaptive is 2.21× slower. | Both CARE-Static and CARE-Adaptive evaluate exactly 80 candidates across 16 steps. Adaptive recomputation incurs a 2.2× runtime investment for superior quality. |
| **Figure 3: Functional Dissociation** | [`audit/experiment4_pair_level.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit/experiment4_pair_level.csv) | **Verified** | Narrative example pairs `[8, 13]` & `[13, 15]` belong to online sequential run, not the 186-pair set. | Within a narrow capability distance band ($|\Delta d| < 10^{-4}$, $N=186$), Oracle KL varies by 4.66× (0.00152 to 0.00710). Valid internal pairs show 4.23× variance. |
| **Aggressive 50% Merging Robustness** | [`experiments/exp15_merging_benchmarks/report.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp15_merging_benchmarks/report.md) | **Verified** | Evaluated on WikiText-2 PPL only; downstream reasoning pending. | At 50% compression, CARE-Adaptive avoids catastrophic perplexity collapse observed in HC-SMoE and MC-SMoE on JetMoE-8B and Phi-3.5-MoE. |
| **Reasoning Retention vs REAP** | [`benchmark/Qwen3-30B-A3B/exp14_reasoning/pilot_reasoning_90exp_report.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/Qwen3-30B-A3B/exp14_reasoning/pilot_reasoning_90exp_report.json) | **Verified** | Pilot sample size of 100 GSM8K / 50 MATH questions. | On Qwen3-30B-A3B at 90 experts, hard structural pruning (REAP) suffers total reasoning collapse (0%), whereas pairwise merging preserves 22%–28% GSM8K accuracy. |

---

## 4. Immediate Concrete Recommendations

1. **Retract or Rewrite Table 4 and Figure 6:**
   - Remove the claim of "30.4% fewer evaluations" and "80.3% wall-clock speedup."
   - Present honest, transparent accounting: both methods use an evaluation budget of 80; CARE-Adaptive requires 666.35s versus 301.63s for CARE-Static (a 2.2× runtime investment that pays off in a 0.90 PPL / 15.9% KL quality advantage).
2. **Clarify Figure 3 Text:**
   - Replace Candidates `[8, 13]` and `[13, 15]` with the verified internal pairs `[18, 37]` ($d=0.002157, \text{KL}=0.007096$) and `[8, 57]` ($d=0.002159, \text{KL}=0.001676$) to preserve absolute numerical consistency with the 186-pair plot.
3. **Clarify Experiment 4 Scope:**
   - State transparently that Model B's geometry represents a transductive low-rank factorization of the physical merge intervention matrix ($q=4$), rather than an observational zero-shot token probe.
4. **Disambiguate Notation:**
   - Replace Precision@$K$ ($K \in \{10, 25, 50\}$) in Table 1 with Precision@Top-$m$ to avoid confusion with the sequential search budget ($K=5$).

---

## 5. Audit Deliverables Index

All detailed investigation reports and raw analytical data have been placed in [`care_scientific_audit_20261014/`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/):

1. [`audit_summary.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/audit_summary.md) — *This document.*
2. [`figure3_reconciliation.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/figure3_reconciliation.md) — *Detailed numerical provenance of the 186-pair subset, root-cause of narrative pair mismatch, and verified internal replacements.*
3. [`static_adaptive_audit.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/static_adaptive_audit.md) — *Forensic trace of `care_static/trajectory.json` logging bug, true wall-clock and evaluation accounting, and first-step analysis.*
4. [`geometry_validation.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/geometry_validation.md) — *Expert-disjoint cross-validation audit, transductive Oracle MDS analysis, dyadic pair clustering, and $N=5$ power limits.*
5. [`random_k_experiment_design.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/random_k_experiment_design.md) — *Ready-to-implement experimental specification for the Matched Random-K control ($K=5$, seeds 101–505).*
6. [`claim_evidence_matrix.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/claim_evidence_matrix.csv) — *Machine-readable claim-to-evidence matrix covering all major paper claims.*

---

## 6. Auditor Compliance Sign-Off

The auditor certifies that:
- [x] No GPU compute was utilized.
- [x] No inference, training, perplexity, or reasoning jobs were launched.
- [x] No running processes on the L40S VM or local environment were interrupted or modified.
- [x] No submitted paper files, source code, raw data, checkpoints, or existing experiment logs were overwritten or modified.
- [x] All generated files were isolated strictly within [`care_scientific_audit_20261014/`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_scientific_audit_20261014/).
