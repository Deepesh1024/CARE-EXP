# CARE-COM Expert Compression Benchmark Summary (OLMoE-1B-7B-0924)

## Scientific Progression (3 Layers of Evidence)

### Layer 1 — Predictive geometry
Experiment 4 demonstrated that capability geometry successfully predicts functional relationships:
- Local baseline rho ≈ 0.480
- Geometry-only rho ≈ 0.750
- Combined rho ≈ 0.815

### Layer 2 — Intervention consequence
v2.1/v2.2 isolated the controlled first-step analysis:
Given the same initial 64-expert model:
- Static candidate [8, 13]: capability distance ≈ 0.0156, actual KL ≈ 0.074
- Adaptive candidate [13, 15]: capability distance ≈ 0.0157, actual KL ≈ 0.046

Conclusion: Candidate capability similarity does not uniquely determine actual intervention damage. Nearly indistinguishable capability distances can correspond to materially different intervention consequences, motivating explicit functional evaluation and adaptive recomputation.

### Layer 3 — Compression benchmark
The final external benchmark establishes that adaptive functional compression yields:
- Competitive PPL with recent expert-merging baselines (Sub-MoE).
- Lower functional divergence than static CARE.
- Substantially lower PPL than HC-SMoE, M-SMoE, and random merging.

## Main Benchmark Performance (PPL)
CARE-Adaptive achieves the lowest PPL at 60 experts (11.54) and remains highly competitive with Sub-MoE at stronger compression (13.81 vs 13.73 at 56 experts), while substantially outperforming HC-SMoE, M-SMoE, and random merging at all evaluated aggressive compression levels.

## Effect of Adaptive Recalculation (Ablation)
Adaptive recomputation consistently reduces cumulative functional divergence relative to the frozen static capability ranking:
- 60 experts: ~17.8% reduction
- 56 experts: ~24.6% reduction
- 48 experts: ~15.9% reduction

## Efficiency Result
At the 48-expert target (25% parameter reduction):
- **Exact evaluation reduction:** ≈ 30.4%
- **Compression-time reduction:** ≈ 80.3%

*Note: In our implementation, adaptive candidate selection required fewer exact intervention evaluations and less wall-clock compression time than Static CARE-COM. Wall-clock times across external implementations (like HC-SMoE) may not be directly comparable because they use different computational procedures and should not be interpreted as an intrinsic theoretical complexity comparison.*

## Notes on Functional KL Data
We measured Perplexity (PPL) across all methods. Functional Cumulative and Marginal KL divergence was only available for methods for which we could directly instrument the intervention trajectory (CARE-Adaptive, CARE-Static, and Random). Missing external KL measurements have been explicitly withheld from visualization to prevent the false implication of zero functional damage.
