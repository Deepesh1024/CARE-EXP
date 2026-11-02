# CARE-COM: Dynamic Functional Representations for MoE Expert Consolidation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.3+](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)

An empirical research framework investigating whether **dynamically updated functional representations** can guide safer expert consolidation in Mixture-of-Experts (MoE) architectures without retraining.

---

## 🔬 Core Research Question

> **Can dynamically updated functional representations guide safer MoE expert consolidation?**

Mixture-of-Experts (MoE) architectures achieve high capacity with sparse computational cost by activating only a fraction of their parameters per token. However, deployment memory requirements scale with total expert count. Consolidating experts via parameter merging often induces catastrophic capability degradation because naive parameter similarity or static routing distributions fail to account for how expert functional geometry shifts as consecutive merges take place.

**CARE-COM** (**CA**pability-**R**epresentation-**E**nhanced **CO**nsolidation of **M**ixture-of-experts) formulates consolidation as an active, state-dependent process:
1. **Dynamic Functional Profiling:** Probing expert activation responses across 10 controlled capability axes and recomputing representations as the model structure evolves ($C_t \in \mathbb{R}^{N_t \times (L \times 10)}$).
2. **Nomination & Filtering:** Identifying the top-$K$ candidate expert pairs with minimal functional distance.
3. **Marginal Damage Selection:** Evaluating candidate merges using local forward KL divergence before committing to physical weight updates, protecting against destructive interference.

---

## 📊 Comprehensive 8-Baseline Benchmark (OLMoE-1B-7B)

All 8 evaluated configurations are categorized into a clear 3-tier taxonomy. Deep compression targets $64 \to 48$ experts ($25\%$ reduction, 16 merge steps) evaluated on WikiText-2 test perplexity (PPL).

| Category | Method | Implementation Status | 48-Expert PPL | PPL Degradation ($\Delta$) | Description |
|:---|:---|:---|:---:|:---:|:---|
| **Reference** | **Uncompressed Base** | `Uncompressed checkpoint` | **10.2804** | baseline | Evaluated `allenai/OLMoE-1B-7B-0924` base model |
| **Proposed** | **CARE-Adaptive** | `Validated implementation` | **19.88 ± 0.92** | **+9.60** | Dynamic functional recomputation + marginal KL selection ($K=5$) across $N=3$ independent seeds |
| **Internal Controls** | **Random-K5** | `Validated implementation / Internal control` | **22.33 ± 0.92** | +12.05 | Random pair nomination with identical marginal KL selection ($K=5$) |
| | **CARE-Static** | `Validated implementation / Internal control` | **24.33 ± 0.87** | +14.05 | Fixed initial capability ranking without dynamic state updates |
| **Expert Merging Baselines** | **MC-SMoE Proxy** | `In-house proxy` | **26.74 ± 0.00** | +16.46 | Magnitude-centric similarity merging proxy |
| | **HC-SMoE Proxy** | `In-house proxy` | **31.00 ± 0.00** | +20.72 | Hierarchical routing-frequency clustering merge proxy |
| **Alternative Expert Reduction** | **Sub-MoE Proxy** | `In-house proxy` | **34.84 ± 0.00** | +24.56 | Frequency-based sub-network selection proxy |
| | **REAP-Prune** | `Faithful reproduction / Structured pruning` | **13.01 ± 0.00** | +2.73 | Router-weighted energy allocation pruning (excises experts without merging) |

### Key Benchmark Findings & Scientific Defensibility

1. **Internal Control Separation:**
   - CARE-Adaptive ($19.88$) outperforms Random-K5 ($22.33$) by **$2.45$ PPL**, confirming that functional capability nomination provides informative structural guidance over random candidate pooling.
   - CARE-Adaptive outperforms CARE-Static ($24.33$) by **$4.45$ PPL**, demonstrating that **dynamic state recomputation** prevents the progressive path-dependence errors observed when freezing initial representations.
2. **Expert Merging Baselines:**
   - Against direct parameter-merging proxies, CARE-Adaptive achieves a **$6.86$ PPL** improvement over MC-SMoE ($26.74$) and an **$11.12$ PPL** improvement over HC-SMoE ($31.00$).
3. **Alternative Expert Reduction (Pruning vs. Merging):**
   - In this zero-shot evaluation, REAP-Prune achieves substantially lower PPL than the evaluated merging methods. This highlights a fundamental trade-off: pruning and parameter merging alter model structure through different mechanisms and should not be interpreted as directly equivalent approaches. Pruning eliminates unrouted parameters, whereas merging attempts to compress distributed knowledge into surviving weights.

---

## 🗺️ Experimental Program & Research Roadmap

The repository preserves the complete scientific evolution across five progressive stages:

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 CARE-COM Research Evolution Program                    │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Stage I: Functional Descriptor Development                             │
  │ • Experiment 1: Univariate Feature Evaluation                          │
  │ • Experiment 1.5: Multivariate Linearization Gap Analysis              │
  │ • Experiment 2: Capability-Aware Feature Engineering                   │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Stage II: Functional Structure and Geometry                            │
  │ • Experiment 3A: Capability Graph Discovery                            │
  │ • Experiment 3B: Capability Geometry Validation                       │
  │ • Experiment 3C: Structural Audit & Trajectories                      │
  │ • Experiment 4: Noise Ceiling & Routing Analysis                      │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Stage III: Functional Applications                                     │
  │ • Experiment 5: Compression Benchmarks & Trajectories                 │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Stage IV: Dynamics and Interventions                                   │
  │ • Experiment 6B: Empirical Laws of Structural Evolution               │
  │ • Experiment 6C: Structural Evolution Modeling                        │
  │ • Experiment 6D: Multi-Directional Intervention Responses             │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Stage V: Cross-Model Verification & Baselines                          │
  │ • Experiment 7B: CARE-COM Failure Analysis (Joint Interaction)        │
  │ • Experiment 7C: Functional Neuron Geometry vs. Residuals             │
  │ • CARE-COM v2.2: Full-Spectrum Nomination & Dimension Alignment       │
  │ • Cross-Model Transfer: OLMoE-1B, JetMoE-8B, Phi-3.5-MoE, Qwen3-30B   │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Model Evaluation Matrix: Validated vs. Exploratory

| Model | Parameter Scale | Expert Topology | Status | Evaluation Summary | Primary Documentation |
|:---|:---|:---|:---|:---|:---|
| **OLMoE-1B-7B-0924** | 1.3B active / 6.9B total | 64 experts / Top-8 | **Validated** | Multi-seed benchmark across all 8 baselines; full PPL & KL metrics | [`reports/CARE_COM_Research_Brief.md`](reports/CARE_COM_Research_Brief.md) |
| **JetMoE-8B** | 2.2B active / 8.5B total | 8 MLP / 8 Attn experts | *Exploratory* | Multi-module dual-router MoE compression | [`benchmark_results/JetMoE-8B/`](benchmark_results/JetMoE-8B/) |
| **Phi-3.5-MoE-instruct** | 6.6B active / 41.9B total | 16 experts / Top-2 | *Exploratory* | Quantized INT4 sequential MoE consolidation | [`benchmark_results/Phi-3.5-MoE/`](benchmark_results/Phi-3.5-MoE/) |
| **Qwen3-30B-A3B** | 3.3B active / 30.0B total | 128 experts / Top-8 | *Exploratory* | Ultra-large MoE single-GPU feasibility study | [`benchmark_results/Qwen3-30B-A3B/`](benchmark_results/Qwen3-30B-A3B/) |

---

## 📂 Repository Architecture

```text
CARE-EXP/
├── README.md                           # Master research overview & benchmark documentation
├── requirements.txt                    # Python runtime dependencies
├── LICENSE                             # MIT License
├── experiments/                        # Core algorithmic implementations and historical runs
│   ├── experiment1/ ... experiment7c/  # Stages I through V experimental pipelines
│   └── care_com_v21/                   # Core CARE-COM v2.1/v2.3 compression engine & evaluator
├── benchmark/                          # Modular benchmarking framework
│   ├── compressors/                    # Standardized compressors (CARE-Adaptive, CARE-Static, Random)
│   ├── OLMoE-1B-7B/                    # Model-specific configs, scripts, and baseline implementations
│   ├── JetMoE-8B/                      # JetMoE-8B benchmarking pipelines
│   ├── Phi-3.5-MoE/                    # Phi-3.5-MoE benchmarking pipelines
│   ├── Qwen3-30B-A3B/                  # Qwen3-30B-A3B benchmarking pipelines
│   ├── run_cross_capability_campaign.py # Multi-probe capability evaluation harness
│   └── run_l40s_benchmark.py           # Single-GPU hardware execution pipeline
├── benchmark_results/                  # Raw trajectories, checkpoints, and evaluations
│   ├── OLMoE-1B-7B/                    # 8-baseline trajectory outputs and metrics
│   ├── JetMoE-8B/                      # Trajectory outputs and metrics
│   ├── Phi-3.5-MoE/                    # Trajectory outputs and metrics
│   └── Qwen3-30B-A3B/                  # Trajectory outputs and metrics
├── reports/                            # Curated scientific briefs and reconciled findings
│   ├── BENCHMARK_L40S/                 # Consolidated L40S benchmark outputs
│   ├── BENCHMARK_MANIFEST.yaml         # Reproducibility job manifest
│   ├── CARE_COM_Research_Brief.pdf     # One-page executive summary brief
│   ├── CARE_COM_TARGETED_EVIDENCE_RECONCILIATION.md # Evidence reconciliation ledger
│   └── CROSS_MODEL_CAPABILITY/         # Cross-model evaluation artifacts
├── results/                            # Historical stage findings, plots, and experiment reports
│   └── exp1/ ... exp6d/                # Empirical outputs, regression tables, and figures
├── scripts/                            # Operational utility and VM validation scripts
│   └── validation/                     # Real-model GPU verification harness
└── tests/                              # Rigorous unit and invariant test suite (100% passing)
    ├── test_physical_merge_engine.py   # Tensor restoration & bitwise identity verification
    ├── test_compression_interface.py   # Standardized compressor API conformance
    ├── test_faithful_baselines.py      # Baseline implementations unit tests
    └── test_v22_forensic_suite.py      # Numerical stability and candidate pool bounding tests
```

---

## ⚡ Quickstart & Reproducibility

### 1. Installation

```bash
git clone https://github.com/Deepesh1024/CARE-EXP.git
cd CARE-EXP

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Verification Suite

Before launching experiments, verify that all core engine invariants, tensor snapshots, and numerical bounds pass on your machine:

```bash
python3 -m unittest discover tests
```

### 3. Running Benchmarks

To execute the compression benchmark on `OLMoE-1B-7B` with a specific compressor:

```bash
# Run CARE-Adaptive compression (48 experts, candidate pool size 5)
python3 benchmark/OLMoE-1B-7B/methods/care_com.py --target_experts 48 --candidate_pool_size 5 --seed 42

# Run Random-K5 control
python3 benchmark/OLMoE-1B-7B/methods/random_merge.py --target_experts 48 --candidate_pool_size 5 --seed 42
```

---

## 🎯 Next Steps & Future Work

While the repository cleanup and baseline restructuring provide a clean foundation, the primary ongoing research priority is:
1. **Strengthening Statistical Power:** Scaling multi-seed evaluations across additional random seeds ($N \ge 10$) to further tighten confidence intervals on the $+2.45$ PPL margin over Random-K5.
2. **Downstream Task Generalization:** Expanding evaluation beyond WikiText-2 perplexity to multi-task benchmark suites (MMLU, GSM8K, ARC-C) under high-sample regimes.
3. **Cross-Architecture Validation:** Scaling validated multi-seed runs from OLMoE-1B-7B to larger architectures (e.g., Qwen3-30B-A3B) on high-memory multi-GPU clusters.

---

## 📄 License & Attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
If you find this work or framework helpful in your research, please cite:

```bibtex
@misc{care_com_2026,
  title  = {CARE-COM: Dynamic Functional Representations for MoE Expert Consolidation},
  author = {Deepesh Kumar Jha and Contributors},
  year   = {2026},
  url    = {https://github.com/Deepesh1024/CARE-EXP.git}
}
```
