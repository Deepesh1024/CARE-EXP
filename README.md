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
  │ 🟢 Experiment 1.5: Multivariate Linearization Gap Analysis            │
  │ 🟢 Experiment 2: Capability-Aware Feature Engineering                 │
  │ 🟢 Experiment 3 (A-C): Capability Graph Discovery & Geometry          │
  │ 🟢 Experiment 4: Noise Ceiling & Routing Analysis                     │
  │ 🟢 Experiment 5: Compression Benchmarks & Trajectories                │
  │ 🟢 Experiment 6 (A-D): Structural Evolution & Interventions           │
  └───────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Key Scientific Reports

All experimental phases are comprehensively documented with mathematical rigor, visual figure embeds, and failure analysis:

| Research Paper / Report | Scope & Content Highlights |
|---|---|
| [**Unified Master Research Report**](./full_report.md) | **Recommended starting point.** Covers the complete scientific journey spanning Experiments 1, 1.5, and 2. |
| [**Experiment 2: Capability-Aware Descriptors**](./results/exp2/report.md) | Formulation, complexity, and dominance of our new pre-merge capability descriptors. |
| [**Experiment 1.5: Linearization Gap**](./results/exp1_5/report.md) | Multivariate capability modeling and the disqualification of oracle-grade features. |
| [**Experiment 1: Univariate Study**](./results/exp1/report.md) | Detailed empirical evaluation of 7 standalone heuristics. |
| [**Experiment 3A: Capability Graph Discovery**](./results/exp3a/experiment3a_report.md) | Capability Graph Discovery. |
| [**Experiment 3B: Capability Geometry**](./results/exp3b/final_report.md) | Capability Geometry Validation (Phase A). |
| [**Experiment 3C: Structural Audit**](./results/exp3c/analysis/analysis_report.md) | Analysis Report & Structural Audit. |
| [**Experiment 4: Noise Ceiling**](./results/exp4/final_report.md) | Final Report for Experiment 4. |
| [**Experiment 5: Compression Benchmarks**](./results/exp5/compression_summary.md) | Experiment 5 Compression Results. |
| [**Experiment 6B: Empirical Laws**](./results/exp6b/EXP6B_FINAL_REPORT.md) | FINAL REPORT for Experiment 6B. |
| [**Experiment 6C: Structural Evolution**](./results/exp6c/EXP6C_FINAL_REPORT.md) | FINAL REPORT for Experiment 6C. |
| [**Experiment 6D: Interventions**](./results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md) | Multi-Directional Intervention Responses. |

---

## 📂 Repository Architecture

```
CARE-EXP/
├── README.md                           # Project landing page & navigation guide
├── experiments/                        # Core algorithmic execution suites
│   ├── experiment1/                    # Univariate evaluation suite
│   ├── experiment1_5/                  # 3-Phase multivariate regression suite
│   ├── experiment2/                    # 7-Phase Descriptor Engineering suite
│   ├── experiment3a/                   # Capability Graph Discovery
│   ├── experiment3b/                   # Capability Geometry Validation
│   ├── experiment3c/                   # Structural Audit
│   ├── experiment4/                    # Noise Ceiling & Routing Analysis
│   ├── exp5/                           # Compression Benchmarks
│   ├── experiment6a/                   # Phase 6A Analysis
│   ├── experiment6b/                   # Empirical Law Analysis
│   ├── experiment6c/                   # Structural Evolution
│   └── experiment6d/                   # Multi-Directional Intervention
└── results/                            # Persistent artifacts, datasets, & reports
    ├── exp1/ ... exp6d/                # Experiment output storage and final markdown reports
```

---

## ⚡ Quick-Start Guide

### 1. Installation & Environment
Ensure you have Python 3.9+ and PyTorch installed, along with our regression and visual evaluation dependencies:

```bash
pip install torch torchvision torchaudio transformers accelerate
pip install scikit-learn xgboost shap pandas numpy scipy matplotlib tabulate
```

### 2. Replicating the Experiments
Each experiment has been containerized into a sequential execution pipeline via a single `run_all.py` or equivalent orchestrator. This allows 1-click reproduction of the full analysis suite, generating metrics, plots, and final reports automatically.

Run any of the following commands from the root directory:

```bash
# Experiment 2: Capability-Aware Descriptors
python3 experiments/experiment2/run_all.py

# Experiment 3A: Capability Graph Discovery
python3 experiments/experiment3a/run_all.py

# Experiment 3B: Capability Geometry Validation
python3 experiments/experiment3b/run_all.py

# Experiment 3C: Structural Audit
python3 experiments/experiment3c/run_all.py

# Experiment 4: Noise Ceiling
python3 experiments/experiment4/run_all.py

# Experiment 5: Compression Benchmarks
python3 experiments/exp5/run_all.py

# Experiment 6B: Empirical Laws
python3 experiments/experiment6b/run_all.py

# Experiment 6C: Structural Evolution
python3 experiments/experiment6c/run_all.py

# Experiment 6D: Multi-Directional Intervention
python3 experiments/experiment6d/run_final.py
```

### 3. Legacy Experiments
To re-verify the baseline algorithms from earlier experiments, you can execute their specific scripts:

```bash
# Experiment 1.5 Pipeline
python3 experiments/experiment1_5/phase1_dataset.py
python3 experiments/experiment1_5/phase2_regression.py
python3 experiments/experiment1_5/phase3_analysis.py
```

---

## 📝 Citation & License
This project is licensed under the MIT License. If you reference our findings, datasets, or theoretical definitions in your research, please cite:

```bibtex
@misc{care_moe_2026,
  title  = {CARE-MoE: Capability-Aware Redundancy Elimination in Mixture-of-Experts Language Models},
  author = {Anonymous and Contributors},
  year   = {2026},
  url    = {https://github.com/anonymous/CARE-EXP}
}
```
