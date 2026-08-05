# CARE-MoE: Capability-Aware Redundancy Elimination in Mixture-of-Experts

![Research Stage](https://img.shields.io/badge/Research_Stage-Exp_1.5_Complete_%7C_Exp_2_Planned-blue.svg)
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
  │    • Findings: Linearization Gap Δ = +0.100; Linear R² = 33.2%.         │
  │    • Decision: Outcome B (Current pre-merge features insufficient).   │
  │    • Documentation: results/exp1_5/report.md                          │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 🟡 Experiment 2: Capability-Aware Feature Engineering [NEXT PHASE]    │
  │    • Status: Architectural Design Finalized                           │
  │    • Targets: Output Magnitude Asymmetry (Δ_mag), Routing JSD,        │
  │               Routing NPMI Co-Activation, & Specialization Entropy.   │
  │    • Objective: Lift linear prediction above tree ceiling (ρ >= 0.59).│
  └───────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Key Scientific Reports

All experimental phases are comprehensively documented with mathematical rigor, visual figure embeds, and failure analysis:

| Research Paper / Report | Scope & Content Highlights |
|---|---|
| [**Unified Master Research Report**](./full_report.md) | **Recommended starting point.** Covers the complete scientific journey spanning Experiments 1 and 1.5, details our **non-injective** scatter discoveries, and establishes the formal mathematical specification for **Experiment 2**. |
| [**Experiment 1: Univariate Study**](./results/exp1/report.md) | Detailed empirical evaluation of 7 standalone heuristics (Weight Distance, Cosine, Activation Sim, Routing Sim, etc.) across calibration sequence budgets ($N=64, 128, 256$) and network layer depths (`first`, `middle`, `last`). |
| [**Experiment 1.5: Linearization Gap**](./results/exp1_5/report.md) | Exhaustive 17-section publication on multivariate capability modeling, disjoint expert partitioning, the disqualification of oracle-grade features, heteroscedastic error analysis, and proof of **Outcome B**. |
| [**Experiment 1.5 Automated Summary**](./results/exp1_5/experiment1_5_report.md) | Executive visual briefing generated directly by `phase3_analysis.py`, showcasing the 6 foundational 300 DPI analytical charts and core Q&A scientific deductions. |

---

## 📂 Repository Architecture

```
CARE-EXP/
├── full_report.md                      # Master unified scientific research paper (Exp 1 & 1.5)
├── README.md                           # Project landing page & navigation guide
│
├── experiments/                        # Core algorithmic execution suites
│   ├── experiment1/                    # Univariate evaluation suite
│   │   ├── CARE_MoE_V3_E1.py           # Token activation calibration & Oracle KL correlation pipeline
│   │   └── plot.py                     # N-segmented scatterplot visualization synthesizer
│   │
│   └── experiment1_5/                  # 3-Phase multivariate regression suite
│       ├── config.py                   # Centralized hyperparameter, deterministic paths & split bounds
│       ├── utils.py                    # Logging formatting and disk safety guardrails
│       ├── phase1_dataset.py           # Disjoint expert partitioner (prevents identity leakage)
│       ├── phase2_regression.py        # OLS, Ridge, LASSO, & XGBoost model training engine
│       └── phase3_analysis.py          # High-resolution figure renderer, SHAP explainer, & reporter
│
└── results/                            # Persistent artifacts, raw datasets, & visual output
    ├── exp1/                           # Experiment 1 output storage
    │   ├── report.md                   # Dedicated Exp 1 academic report
    │   ├── output.json                 # Raw evaluation checkpoints (18,644 expert pairs in OLMoE-1B-7B)
    │   └── {64,128,256}_segmented/     # Layer-stratified univariate scatterplots
    │
    └── exp1_5/                         # Experiment 1.5 output storage
        ├── report.md                   # Complete 640-line academic research paper
        ├── experiment1_5_report.md     # Automated Phase 3 visual summary
        ├── figures/                    # High-resolution 300 DPI publication charts
        └── models/                     # Serialized scalers, regression equations, & XGBoost trees (.pkl)
```

---

## ⚡ Quick-Start Guide

### 1. Installation & Environment
Ensure you have Python 3.9+ and PyTorch installed, along with our regression and visual evaluation dependencies:

```bash
pip install torch torchvision torchaudio transformers accelerate
pip install scikit-learn xgboost shap pandas numpy scipy matplotlib
```

### 2. Replicating Experiment 1.5 (Multivariate Pipeline)
To re-verify the **$+0.100$ Linearization Gap** and train all 12 experimental regression models from scratch using the raw checkpoint data in `results/exp1/output.json`:

```bash
# Phase 1: Ingest raw data, apply Seq_Len=256 filters, & generate strict disjoint split
python experiments/experiment1_5/phase1_dataset.py

# Phase 2: Train OLS, Ridge, LASSO, and XGBoost across Variants A, B, and C
python experiments/experiment1_5/phase2_regression.py

# Phase 3: Calculate SHAP values, generate 300 DPI figures, & build automated summary
python experiments/experiment1_5/phase3_analysis.py
```
*Outputs, trained checkpoints, and new graphs will automatically be saved into `results/exp1_5/`.*

### 3. Running Experiment 1 (Univariate Calibration)
To re-run the calibration generation and univariate correlation calculations on OLMoE-1B-7B:

```bash
python experiments/experiment1/CARE_MoE_V3_E1.py
python experiments/experiment1/plot.py
```

---

## 🎯 What Lies Ahead: Experiment 2

With **Outcome B** scientifically established—confirming that linear formulations using conventional pre-merge heuristics fall below the required similarity threshold ($\rho_{\text{linear}} = 0.578 < 0.80$, $\Delta = +0.100$)—our immediate next focus is **Experiment 2 (Capability-Aware Feature Engineering)**. 

We will design and test:
1. **Output Magnitude Asymmetry ($\Delta_{\text{mag}}$)**
2. **Routing Jensen-Shannon Divergence ($\text{JSD}$)**
3. **Routing NPMI (Co-Activation Dependency)**
4. **Expert Specialization Entropy ($H_{\text{spec}}$)**

Our success criterion is achieving a linear regression Spearman rank correlation that equals or surpasses the existing nonlinear decision tree ceiling: $\rho_{\text{linear}} \ge 0.593$.

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
