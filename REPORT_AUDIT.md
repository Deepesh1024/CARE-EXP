# Final Scientific Audit: Interpretability as a Science

This document records the systematic scientific audit and resulting corrections applied across the `CARE-MoE/Experiments-V3` repository in preparation for the NeurIPS 2026 "Interpretability as a Science" workshop submission.

The core objective was to bound all claims strictly to empirical evidence, explicitly avoiding unproven mathematical overclaims ("manifold"), ensuring transparency regarding negative/mixed results (Precision@K inversion), and clearly categorizing findings into Observational, Predictive, and Interventional claims.

## 1. Global Terminology Purge

A systematic search-and-replace was performed across all reports to neutralize hyperbole.

| Original Term / Concept | Corrected Term / Bound | Justification |
| :--- | :--- | :--- |
| `Manifold` (unbounded) | `Structured functional geometry` | We have not proven infinite differentiability or local Euclidean topology in a mathematical sense; we observe predictive low-dimensional geometric structure. |
| `Universally`, `Universal` | `Broadly`, `Broad` | Avoids claims of zero exceptions. |
| `Causal proof` | `Interventional evidence` | Interventions support a geometric hypothesis but do not satisfy formal causal DAG constraints. |
| `Fundamentally proves`, `Guarantees` | `Strongly supports`, `Indications` | Epistemically appropriate bounds for empirical science. |

## 2. Granular Experiment Corrections

### Experiment 3A (Graph Architecture)
- **Original Hypothesis/Claim:** The capability space possesses a modularity $Q$ that is "significantly greater than random chance."
- **Corrected Claim (Observational):** Expert relationships exhibit non-uniform community structure, with the strongest modular organization appearing in the middle layer. 
- **Justification:** Permutation/null statistics required to establish true significance relative to an ER graph were lacking in the report, so the statement was downgraded to an exploratory topological observation.

### Experiment 3B (Geometry Validation)
- **Original Claim:** "We prove a manifold structure" or "the capability space is a manifold."
- **Corrected Claim (Predictive):** Expert capabilities exhibit a low-dimensional, continuously structured functional geometry. 
- **Justification:** Successful MDS embedding and holdout validation is necessary but not sufficient to mathematically prove a globally smooth manifold.

### Experiment 3C (Evolution & Topology)
- **Original Claim:** "Universal conservation of topology" and "universal expert evolution."
- **Corrected Claim (Observational):** Functional relationships remain structured across checkpoints as experts undergo continuous differentiation. We specify the *layer-dependent* trajectories, noting the middle layer's U-shaped redundancy bottleneck.
- **Justification:** Topology is not strictly rigid; it evolves, and layers behave distinctly. The U-shaped redundancy curve was documented but obscured by "universal" language.

### Experiment 4 (Functional Merge Landscape)
- **Original Claim:** "Geometry massively dominates standard local heuristics. Combining local features with the geometric prior (Model C) creates a highly robust engine..."
- **Corrected Claim (Predictive):** Geometric distance provides substantial predictive information. However, in the highly selective top-$K$ regime ($K=10$, $K=25$), the pure geometry model (B) actually *outperforms* the combined CARE descriptor (C). Model C only becomes strongest at larger $K$ ($K=50$).
- **Justification:** The original text completely ignored a negative result (the Precision@K inversion). Local features are budget-dependent and can actually degrade performance in highly selective, top-$K$ regimes.

### Experiment 6D (Directional Intervention)
- **Original Claim:** "We proved the manifold" via directional resistance.
- **Corrected Claim (Interventional):** Controlled interventions provide evidence that functional responses are direction-dependent, magnitude-dependent, and approximately linear in the tested low-alpha regime. 
- **Justification:** The resistance to orthogonal shifts strongly *supports* a local geometric model, but does not *prove* a globally valid continuous manifold.

## 3. Generalization Limits Added

A formal limitation block was prominently added to the final overarching conclusion in `full_report.md`:

> **Limitation:** All experiments were conducted on a single open-source model architecture (`OLMoE-1B-7B-0924`). While the internal statistics are highly robust across random seeds and cross-validation folds, we do not claim that these specific geometric properties generalize to all MoE architectures. Future work is required to determine whether these topological phenomena are broad features of MoE training or specific to this model's routing mechanism.

## Conclusion of Audit
The entire repository is now scientifically bounded. The claims reflect a mature, defensible scientific posture suitable for peer review. No new data was invented, and no negative results were obscured.
