# Claim to Evidence Map

**Claim 1: Existing MoE compression commonly treats expert redundancy as a static relation.**
- Status: SUPPORTED
- Source: LITERATURE_AUDIT.md (Re-uses static clusters)
- Main/Appendix: Main (Introduction)

**Claim 2: Functional geometry contains useful information about merge-induced damage.**
- Status: SUPPORTED
- Source: audit/experiment4_pair_level.csv (Spearman rho 0.7657)
- Proposed Figure: fig2_geometry_damage
- Main/Appendix: Main

**Claim 3: Functional similarity does not uniquely determine intervention consequence.**
- Status: SUPPORTED
- Source: audit/experiment4_pair_level.csv (186 pairs with dist ~ median have large variance)
- Proposed Figure: fig3_similarity_vs_consequence
- Caveat: Does not mean geometry is useless, just insufficient for exact ordering.
- Main/Appendix: Main

**Claim 4: Physical intervention changes the functional state of the remaining expert population.**
- Status: SUPPORTED
- Source: audit/static_vs_adaptive_validation.md (Recomputation yields 15-24% KL reduction)
- Proposed Figure: fig4_static_vs_adaptive
- Main/Appendix: Main

**Claim 5: Therefore compression can be formulated as a sequential functional intervention problem.**
- Status: SUPPORTED
- Main/Appendix: Main (Conceptual conclusion of Section 3)

**Claim 6: Adaptive CARE-COM outperforms static under matched conditions.**
- Status: SUPPORTED
- Source: audit/static_vs_adaptive_validation.md
- Main/Appendix: Main

**Claim 7: External expert-compression baselines provide practical validation.**
- Status: SUPPORTED
- Source: benchmark_results/summary/results.csv (Outperforms Random, HC-SMoE, M-SMoE, competitive with Sub-MoE)
- Proposed Figure: fig5_external_benchmark
- Main/Appendix: Main

**Claim 8: Exact intervention evaluation remains the major computational bottleneck.**
- Status: SUPPORTED
- Source: benchmark_results/summary/results.csv
- Proposed Figure: fig6_efficiency (30% reduction in evals)
- Main/Appendix: Main
