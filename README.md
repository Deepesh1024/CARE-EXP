# CARE-MoE: Capability-Aware Redundancy Elimination in Mixture-of-Experts

![Research Stage](https://img.shields.io/badge/Research_Stage-Exp_2_Complete_%7C_Canonical_Release-blue.svg)
![Target Architecture](https://img.shields.io/badge/Model-OLMoE--1B--7B-purple.svg)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)

> **An explainable, lightweight research framework for quantifying **expert capability**, redundancy, and mergeability in Mixture-of-Experts (MoE) language models without expensive oracle forward-pass evaluations.**

---

## 🔬 Core Research Vision

As **Mixture-of-Experts (MoE)** architectures dominate large-scale model scaling, compressing layer parameters via expert merging becomes essential for deployment efficiency. However, conventional pruning and averaging heuristics struggle because they rely either on basic **parameter distance**—which destroys model performance—or iterative **oracle evaluation** (re-running forward passes after each test merge), which is computationally **intractable** at scale.

The **CARE-MoE** research program is dedicated to solving this problem not by proposing ad-hoc compression hacks, but by discovering mathematically rigorous, pre-merge descriptors that model **latent expert capability**, **domain specialization**, and **operational redundancy**.

---

## 🗺️ Research Roadmap & Current Progress

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │                 CARE-MoE Research Evolution Roadmap                 │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 🟢 Experiment 1: Univariate Feature Evaluation                        │
  │    • Status: Completed & Published                                    │
  │    • Findings: All 7 individual similarity features fail (|ρ| < 0.2). │
  │    • Discovery: Capability is an emergent, latent network property.   │
  │    • Documentation: results/exp1/report.md                            │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 🟢 Experiment 1.5: Multivariate Linearization Gap Analysis            │
  │    • Status: Completed & Published                                    │
  │    • Innovation: Enforced strict disjoint expert split (No Leakage).  │
  │    • Discovery: Purged oracle-grade features (CE_Delta, L2_Drift).    │
  │    • Findings: Gap Δ = +0.109; Tree Test R² = -50.7% (Catastrophic).  │
  │    • Decision: Outcome B (Current pre-merge features insufficient).   │
  │    • Documentation: results/exp1_5/report.md                          │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 🟢 Experiment 2: Capability-Aware Feature Engineering [COMPLETE]      │
  │    • Status: Completed & Published                                    │
  │    • Innovation: Engineered 4 pre-merge capability descriptors        │
  │                  (NPMI_routing, JSD_routing, Usage_Asym, Spec_Diff).  │
  │    • Discovery: Routing_NPMI_Proxy is #1 in XGBoost (15.98% gain)!    │
  │    • Findings: Linearization gap is layer-localized (Late layer linear│
  │                ρ = 0.835, gap < 0.02).                                │
  │    • Prescription: Trees for early gating, linear models for deep     │
  │                    layers.                                            │
  │    • Documentation: results/exp2/report.md                            │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 🟣 Experiment 3: Layer-Adaptive Compression Engine [PLANNED]          │
  │    • Target: End-to-end LLM decoding efficiency benchmarking.         │
  └───────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Key Scientific Reports

All experimental phases are comprehensively documented with mathematical rigor, visual figure embeds, and failure analysis:

| Research Paper / Report | Scope & Content Highlights |
|---|---|
| [**Unified Master Research Report**](./full_report.md) | **Recommended starting point.** Covers the complete scientific journey spanning Experiments 1, 1.5, and the newly completed Experiment 2 descriptor engineering findings. |
| [**Experiment 2: Capability-Aware Descriptors**](./results/exp2/report.md) | Exhaustive 26-section scientific publication documenting the formulation, complexity, and dominance of our new pre-merge capability descriptors (including `Routing_NPMI_Proxy`, the #1 predictive feature), layer-localized non-linearity, and deployment protocols. |
| [**Experiment 1.5: Linearization Gap**](./results/exp1_5/report.md) | Exhaustive publication on multivariate capability modeling, disjoint expert partitioning, the disqualification of oracle-grade features, heteroscedastic error analysis, and proof of **Outcome B**. |
| [**Experiment 1: Univariate Study**](./results/exp1/report.md) | Detailed empirical evaluation of 7 standalone heuristics across calibration sequence budgets ($N=64, 128, 256$) and network layer depths. |

---

## 📂 Repository Architecture

```
CARE-EXP/
├── full_report.md                      # Master unified scientific research paper (Exp 1 & 1.5 & 2)
├── README.md                           # Project landing page & navigation guide
│
├── experiments/                        # Core algorithmic execution suites
│   ├── experiment1/                    # Univariate evaluation suite
│   │   ├── CARE_MoE_V3_E1.py           # Token activation calibration & Oracle KL correlation pipeline
│   │   └── plot.py                     # N-segmented scatterplot visualization synthesizer
│   │
│   ├── experiment1_5/                  # 3-Phase multivariate regression suite
│   │   ├── config.py                   # Centralized hyperparameter, deterministic paths & split bounds
│   │   ├── utils.py                    # Logging formatting and disk safety guardrails
│   │   ├── phase1_dataset.py           # Disjoint expert partitioner (prevents identity leakage)
│   │   ├── phase2_regression.py        # OLS, Ridge, LASSO, & XGBoost model training engine
│   │   └── phase3_analysis.py          # High-resolution figure renderer, SHAP explainer, & reporter
│   │
│   └── experiment2/                    # 7-Phase Capability-Aware Descriptor Engineering suite
│       ├── run_all.py                  # Master sequential execution pipeline orchestrator
│       ├── phase0_audit.py             # Feature eligibility registry and oracle exclusion verification
│       ├── phase05_residuals.py        # Residual diagnostic failure mapping of legacy baselines
│       ├── phase075_correlation.py     # Multicollinearity and VIF diagnostic analysis
│       ├── phase1_descriptors.py       # Algorithmic generation of NPMI, JSD, Usage Asymmetry, Spec Diff
│       ├── phase2_diagnostics.py       # Univariate and orthogonal evaluation scatterplots
│       ├── phase3_regression.py        # Model benchmarking suite (Variants A, B, C across hypothesis classes)
│       ├── phase4_interpretability.py  # SHAP trees, LASSO L1 weights, and OOD permutation importance
│       ├── phase5_ablation.py          # Leave-One-Out marginal feature value ranking
│       └── phase6_gap.py               # Linearization Gap comparison, bootstrap p-values, & within-layer analysis
│
└── results/                            # Persistent artifacts, raw datasets, & visual output
    ├── exp1/                           # Experiment 1 output storage
    ├── exp1_5/                         # Experiment 1.5 output storage
    └── exp2/                           # Experiment 2 output storage (metrics.json, plots/, tables/, models/, report.md)
```

---

## ⚡ Quick-Start Guide

### 1. Installation & Environment
Ensure you have Python 3.9+ and PyTorch installed, along with our regression and visual evaluation dependencies:

```bash
pip install torch torchvision torchaudio transformers accelerate
pip install scikit-learn xgboost shap pandas numpy scipy matplotlib tabulate
```

### 2. Replicating Experiment 2 (Capability-Aware Descriptors)
To execute the full 7-phase pipeline, compute the four new capability descriptors, train all model variants, and generate publication charts:

```bash
# Execute master orchestrator (completes all phases in under 20 seconds)
python3 experiments/experiment2/run_all.py
```
*All generated metrics, trained model binaries (`.pkl`), feature importance CSVs, and 300 DPI analytical charts are automatically deposited into `results/exp2/`.*

### 3. Replicating Experiment 1.5 (Multivariate Pipeline)
To re-verify the baseline **$+0.0995$ Linearization Gap** and train the legacy regression suites:

```bash
python3 experiments/experiment1_5/phase1_dataset.py
python3 experiments/experiment1_5/phase2_regression.py
python3 experiments/experiment1_5/phase3_analysis.py
```

---

## 🎯 What Lies Ahead: Experiment 3

With **Experiment 2 completed and published**—proving that our new **`Routing_NPMI_Proxy`** descriptor controls nearly 16% of non-linear tree gain and discovering that the Linearization Gap is concentrated almost entirely in initial transformer layers (while deeper layers achieve super-linear convergence $\rho > 0.83$ with regularized ridge models)—our next frontier is **Experiment 3 (Layer-Adaptive Compression Deployment Engine)**.

We will build and benchmark:
1. **Runtime Layer-Adaptive Pruning:** Implementing hybrid compression routines that invoke lightweight trees for early gating layers and fast linear scoring for deep layers.
2. **Dynamic Calibration Sweeps:** Validating descriptor invariance under domain-shifted multi-lingual and code generation token distributions.

---

## 📝 Citation & License
This project is licensed under the MIT License. If you reference our findings, datasets, or theoretical definitions in your research, please cite:

```bibtex
@misc{care_moe_2026,
  title  = {CARE-MoE: Capability-Aware Redundancy Elimination in Mixture-of-Experts Language Models},
  author = {Deepesh Kumar Jha and Contributors},
  year   = {2026},
  url    = {https://github.com/Deepesh1024/CARE-EXP}
}
```
