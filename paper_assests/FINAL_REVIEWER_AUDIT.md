# Final Reviewer Audit

1. **What is the strongest contribution?** 
The sequential functional intervention framework that explicitly accounts for topological drift, consistently reducing KL divergence relative to static clustering.

2. **What is the weakest major claim?** 
Wall-clock efficiency comparisons against external baselines. Our implementation evaluates fewer candidates, but raw time is hardware/implementation dependent.

3. **What evidence is genuinely novel?** 
The empirical proof (Figure 3) that identical capability distances yield materially divergent functional damages, proving static geometric clustering is insufficient.

4. **What evidence is merely engineering?** 
The benchmark harness and integration of external methods.

5. **What result is most convincing?** 
The controlled Static vs Adaptive ablation, strictly isolating capability recomputation.

6. **What result is easiest for a reviewer to attack?** 
PPL scaling laws (we only evaluated up to 48 experts). Reviewers might ask how this scales to 8 experts.

7. **What baseline comparison is weakest?** 
Sub-MoE, because it is a structural pruning baseline (reduces actual parameters), making it fundamentally a different class than routing-merge methods.

8. **Which claims should be softened?** 
"Compression time is 80% lower" -> "In our controlled implementation, required evaluations and empirical runtime dropped by 80%".

9. **Which claims should be removed?** 
Any implication that capability distance is not useful. It is very useful ($ho=0.76$), just not sufficient.

10. **What experiment would most improve the paper if time allowed?** 
Testing on a non-OLMoE architecture (e.g., Mixtral) to prove generality.

11. **What figure would most improve reviewer understanding?** 
Figure 1 (Method Overview) cleanly distinguishing static clustering from adaptive intervention.

12. **What table is essential?** 
Table 3 (External Benchmark).

13. **What can safely go to appendix?** 
Table 4 (Efficiency), Exp 7 negative results.

14. **What could trigger a reproducibility concern?** 
The dataset chunks used for calibration. We specify WikiText-2 sequential chunks to alleviate this.

15. **What could trigger an anonymity/compliance problem?** 
Accidentally leaving HuggingFace tokens or GitHub paths in the code (fixed).

16. **Are there any hidden protocol mismatches?** 
None remain. All final data relies strictly on the `v3` uniform benchmark.

17. **Is the adaptive-vs-static comparison genuinely controlled?** 
Yes. Same `M_64`, same merge operator, same calibration chunks.

18. **Does the evidence support "sequential functional intervention" as the central framing?** 
Yes, because intervention demonstrably changes the topology, requiring sequential re-evaluation.

19. **Does the evidence support the claim that geometry is useful but insufficient?** 
Yes (Figure 2 vs Figure 3).

20. **What would a skeptical ICLR reviewer likely ask?** 
"Does CARE-COM scale to massive models (8x22B), and what is the exact wall-clock overhead compared to training-free clustering?"
