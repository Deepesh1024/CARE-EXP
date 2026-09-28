# CARE-COM Final Submission Claims

## Claims supported by current experiments
- CARE-Adaptive achieves the lowest Perplexity (PPL) at 60 experts and remains competitive with Sub-MoE at stronger compression, while substantially outperforming HC-SMoE, M-SMoE, and random merging at the evaluated aggressive compression levels.
- Adaptive recomputation consistently reduces cumulative functional divergence relative to the frozen static capability ranking.
- In our implementation, adaptive candidate selection required approximately 30% fewer exact intervention evaluations and approximately 80% less wall-clock compression time than Static CARE-COM at the 48-expert target.
- Predictive Geometry (Experiment 4): Capability geometry successfully predicts functional relationships (Combined rho ≈ 0.815 vs Local baseline rho ≈ 0.480).
- Intervention consequence (v2.1/v2.2): Candidate capability similarity does not uniquely determine actual intervention damage; nearly indistinguishable capability distances can correspond to materially different intervention consequences, thereby motivating explicit functional evaluation and adaptive recomputation.

## Claims that require qualification
- CARE does not beat Sub-MoE at every compression level.
- KL is unavailable for some external methods.
- Runtime is implementation-dependent and should not be interpreted as a direct algorithmic complexity comparison across methods.
- Figure 7 is descriptive, not a predictive validation.

## Claims we must NOT make
- "CARE beats all baselines"
- "CARE achieves the best PPL at every compression ratio."
- "CARE is SOTA"
- "CARE proves its method is superior to all existing methods."
- "Capability distance perfectly predicts intervention damage"
- "CARE has zero functional damage for external methods"
- "CARE automatically knows when the model becomes unusable"
