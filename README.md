# Capability in Mixture-of-Experts: Functional State, Geometry, and Evolution

> **An experimental research framework for representing, analyzing, and studying the functional capability of Mixture-of-Experts (MoE) experts.**

---

## 🔬 Core Research Vision

As Mixture-of-Experts (MoE) architectures grow in prevalence, understanding how individual experts function remains a challenge. Conventional analyses often rely on basic parameter-space similarity or simple token routing frequency, which fail to capture the complex functional contributions of individual experts.

The core research question of this repository is: **How can we operationally represent what an MoE expert functionally contributes, independently of parameter-space similarity and routing frequency?**

This repository approaches this by defining **Capability** as a functional representation (or state) of an MoE expert. Capability is modeled empirically through the expert's activation responses to controlled probes across diverse semantic axes. By establishing this functional state, the research investigates the geometric and structural relationships between experts over the course of training and intervention.

---

## 🧭 Research Scope

It is important to clearly distinguish the primary scientific questions investigated in this repository from their downstream applications.

**Primary Scientific Questions:**
- **Functional Representation:** How do we accurately model the functional capability of an expert?
- **Functional Geometry:** What is the geometric structure of the capability space?
- **Expert Specialization:** How do experts cluster and specialize within this space?
- **Evolution:** How do these structures evolve over the course of model training?
- **Dynamics & Interventions:** How does expert capability respond to controlled architectural modifications?

**Downstream Applications:**
- Redundancy analysis and detection
- Expert similarity scoring
- Layer-adaptive mergeability
- Model compression and deployment efficiency

While redundancy and compression are significant applications evaluated in this work, they are downstream consequences of the broader capability-based representation, not the sole purpose of the framework.

---

## 🗺️ Experimental Roadmap

The repository structures the research into four progressive stages.

```text
  ┌───────────────────────────────────────────────────────────────────────┐
  │                 CARE-MoE Research Evolution Roadmap                 │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Stage I: Functional Descriptor Development                            │
  │ 🟢 Experiment 1: Univariate Feature Evaluation                        │
  │ 🟢 Experiment 1.5: Multivariate Linearization Gap Analysis            │
  │ 🟢 Experiment 2: Capability-Aware Feature Engineering                 │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Stage II: Functional Structure and Geometry                           │
  │ 🟢 Experiment 3A: Capability Graph Discovery                          │
  │ 🟢 Experiment 3B: Capability Geometry Validation                      │
  │ 🟢 Experiment 3C: Structural Audit & Trajectories                     │
  │ 🟢 Experiment 4: Noise Ceiling & Routing Analysis                     │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Stage III: Functional Applications                                    │
  │ 🟢 Experiment 5: Compression Benchmarks & Trajectories                │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Stage IV: Dynamics and Interventions                                  │
  │ 🟢 Experiment 6B: Empirical Laws of Structural Evolution              │
  │ 🟢 Experiment 6C: Structural Evolution Modeling                       │
  │ 🟢 Experiment 6D: Multi-Directional Intervention Responses            │
  └───────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Key Scientific Reports

Each experiment is comprehensively documented with formal methodology, numerical findings, and failure analysis. 

| Stage | Experiment | Research Question & Role in Program | Final Report |
|---|---|---|---|
| **I** | **Exp 1** | *Can standalone heuristics predict capability?* Establishes the baseline failure of basic metrics. | [results/exp1/report.md](./results/exp1/report.md) |
| **I** | **Exp 1.5** | *Can multivariate linear combinations predict capability?* Demonstrates the linearization gap. | [results/exp1_5/report.md](./results/exp1_5/report.md) |
| **I** | **Exp 2** | *How do we engineer accurate pre-merge capability descriptors?* Formulates the core non-linear descriptors (e.g., NPMI proxy). | [results/exp2/report.md](./results/exp2/report.md) |
| **II** | **Exp 3A** | *How are experts clustered functionally?* Investigates capability graph topologies and community boundaries. | [results/exp3a/experiment3a_report.md](./results/exp3a/experiment3a_report.md) |
| **II** | **Exp 3B** | *What is the geometric shape of the capability space?* Validates the functional geometry of the representations. | [results/exp3b/final_report.md](./results/exp3b/final_report.md) |
| **II** | **Exp 3C** | *How do experts evolve structurally over training?* Conducts a longitudinal structural audit across model checkpoints. | [results/exp3c/analysis/analysis_report.md](./results/exp3c/analysis/analysis_report.md) |
| **II** | **Exp 4** | *What is the noise ceiling of routing-based prediction?* Establishes the upper bound of capability prediction from routing behavior. | [results/exp4/final_report.md](./results/exp4/final_report.md) |
| **III** | **Exp 5** | *How well do capability descriptors perform in actual compression?* Benchmarks the framework in downstream expert merging scenarios. | [results/exp5/compression_summary.md](./results/exp5/compression_summary.md) |
| **IV** | **Exp 6B** | *Are there empirical laws governing structural evolution?* Investigates scaling and layer-wise evolutionary patterns. | [results/exp6b/EXP6B_FINAL_REPORT.md](./results/exp6b/EXP6B_FINAL_REPORT.md) |
| **IV** | **Exp 6C** | *How do specific token environments shape structural evolution?* Models the dynamics of capability under controlled semantic exposure. | [results/exp6c/EXP6C_FINAL_REPORT.md](./results/exp6c/EXP6C_FINAL_REPORT.md) |
| **IV** | **Exp 6D** | *How does capability respond to targeted interventions?* Maps the causal-adjacent responses of the network to external modifications. | [results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md](./results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md) |

---

## 📂 Repository Architecture

```text
CARE-EXP/
├── README.md                           # Project research overview & execution guide
├── experiments/                        # Execution suites for all research stages
│   ├── experiment1/                    # Univariate heuristic evaluation
│   ├── experiment1_5/                  # Multivariate regression suite
│   ├── experiment2/                    # Capability-aware descriptor engineering
│   ├── experiment3a/                   # Capability graph discovery pipeline
│   ├── experiment3b/                   # Geometry validation suite
│   ├── experiment3c/                   # Longitudinal checkpoint structural audit
│   ├── experiment4/                    # Noise ceiling and routing analysis
│   ├── exp5/                           # Downstream compression benchmarks
│   ├── experiment6b/                   # Empirical law analysis pipeline
│   ├── experiment6c/                   # Structural evolution modeling
│   └── experiment6d/                   # Multi-directional intervention testing
└── results/                            # Persistent artifacts, datasets, & markdown reports
    ├── exp1/ ... exp6d/                # Experiment outputs (metrics, plots, final reports)
```

---

## ⚡ Reproducibility & Execution

The experiments in this repository consist of sequential Python pipelines that perform model inference, metric extraction, and subsequent analysis.

### 1. Hardware & Environment Requirements
Executing these experiments natively requires adequate hardware for PyTorch forward passes and model instantiation.
- **Compute:** A CUDA-compatible GPU is required for experiments that load language model checkpoints. Some experiments load `allenai/OLMoE-1B-7B-0924` (or similar scales).
- **Storage:** Adequate local storage for HuggingFace model cache weights and generated dataset artifacts.
- **Environment:** Python 3.9+ is recommended. 

Install the exact requirements via:
```bash
pip install -r requirements.txt
```

### 2. Executing Research Pipelines
Each experimental suite contains a central orchestrator script (typically `run_all.py`). These scripts coordinate data parsing, statistical modeling, plotting, and report generation in sequence.

Run any of the following orchestrators from the root directory to reproduce the findings:

**Stage I:**
- `python3 experiments/experiment2/run_all.py`
- *Legacy (Exp 1.5):* Execute `phase1_dataset.py`, `phase2_regression.py`, and `phase3_analysis.py` sequentially in `experiments/experiment1_5/`.

**Stage II:**
- `python3 experiments/experiment3a/run_all.py`
- `python3 experiments/experiment3b/run_all.py`
- `python3 experiments/experiment3c/run_all.py`
- `python3 experiments/experiment4/run_all.py`

**Stage III:**
- `python3 experiments/exp5/run_all.py`

**Stage IV:**
- `python3 experiments/experiment6b/run_all.py`
- `python3 experiments/experiment6c/run_all.py`
- `python3 experiments/experiment6d/run_final.py`

### 3. Expected Outputs
Upon completion, each `run_all.py` script will automatically populate its respective `results/exp[X]/` directory with serialized models (`.pkl`), raw telemetry (`.csv`, `.json`), generated matplotlib charts (`.png`), and a freshly compiled markdown report.

---

## 📝 Citation & License
This project is licensed under the MIT License. If you reference our definitions, empirical datasets, or analytical pipelines in your research, please cite:

```bibtex
@misc{care_moe_2026,
  title  = {Capability in Mixture-of-Experts: Functional State, Geometry, and Evolution},
  author = {Anonymous and Contributors},
  year   = {2026},
  url    = {https://github.com/anonymous/CARE-EXP}
}
```
