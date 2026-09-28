# Repository Audit

A. **Repository structure:**
Contains `results/` with past experiments, `benchmark_results/` with final v3 benchmark data, `audit/` with final reconstructed pairs and diagnostics, and `paper_assests/`.

B. **Relevant files:**
`audit/experiment4_pair_level.csv`, `audit/static_vs_adaptive_validation.md`, `benchmark_results/summary/results.csv`, `benchmark/aggregate_results.py`.

C. **Experiment inventory:**
- Exp 4: Geometry vs Damage (Final)
- Exp 5, 7A, 7B, 7C: Failed/Superseded
- v2.2: Static vs Adaptive Ablation (Final)
- v3: External Benchmark (Final)

D. **Which experiments are reproducible:**
Exp 4 (reconstructed), v2.2, v3 benchmark.

E. **Which results are final:**
v2.2 ablation and v3 benchmark results.

F. **Which results are exploratory:**
Exp 4 was initially exploratory but serves as the foundational geometry proof.

G. **Which results are invalid/incompatible:**
Exp 7A-7C are superseded/invalid.

H. **Which results are suitable for the main paper:**
Exp 4 geometry mapping, v2.2 static vs adaptive KL/PPL, v3 benchmark.

I. **Which results belong only in appendix:**
Failed Exp 7 mechanisms, full benchmark timing tables.

J. **Missing evidence:**
True hardware-agnostic theoretical scaling comparisons against external baselines.

K. **Potential bugs:**
Caught and fixed the NaN-to-zero plotting bug for external baselines.

L. **Protocol inconsistencies:**
v1 used 12-layer subset, v3 uses full 16-layer. We explicitly do NOT mix them.

M. **Claims that cannot currently be supported:**
"CARE-COM universally beats all baselines" (Sub-MoE occasionally has slightly better PPL). "Capability distance perfectly predicts intervention damage" (it doesn't, variance is high).
