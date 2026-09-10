# Experiment 7B: CARE-COM Merge Failure Analysis and Joint Functional Interaction

> **Part of the CARE-MoE (Capability-Aware Redundancy Elimination in Mixture-of-Experts) Research Program.**

---

## 🔬 Central Research Question

> **"When CARE-COM predicts that an expert pair is safe to merge, and that prediction is wrong, does non-additive joint functional interaction explain the failure?"**

CARE-COM identifies expert pairs that appear safe to merge using its capability-based geometry and pre-merge descriptors. However, two experts may individually appear functionally unimportant or compatible while jointly supporting capability that is uncaptured by pre-merge predictions.

This experiment investigates whether non-additive joint functional interaction ($I(i, j)$) between experts explains the prediction error ($E(i, j)$) of CARE-COM.

---

## ⚖️ Core Hypotheses & Pre-Registered Decision Gates

### Hypotheses
- **$H_0$:** Joint functional interaction does not meaningfully explain CARE-COM merge prediction errors ($\rho(I, E) < 0.40$ or $95\%\text{ CI} \le 0$).
- **$H_1$:** Pairs with stronger non-additive joint functional interaction exhibit larger CARE-COM merge prediction errors ($\rho(I, E) \ge 0.40$ with $95\%\text{ CI} > 0$).

### Decision Gates
1. **GATE 1 (Baseline Predictive Utility):**
   - Does existing CARE-COM score predict actual merge damage ($\rho(D_{pred}, D_{actual}) \ge 0.30, p < 0.05$)?
   - **YES:** Proceed to failure analysis (Gate 2).
   - **NO:** Report that CARE-COM has no predictive utility on this benchmark (critical baseline failure).
2. **GATE 2 (Interaction Predictive Power):**
   - Does joint interaction predict merge prediction error ($\rho(I, E) \ge 0.40$ and bootstrap 95% CI strictly positive)?
   - **YES $\implies$ BUILD:** Non-additive interaction confirmed as an explanatory mechanism. Investigate cheap observable proxies.
   - **NO $\implies$ KILL:** Falsification confirmed. Terminate interaction-based additions to CARE-COM.

---

## 🗺️ Experimental Pipeline Stages

The experiment is decomposed into clean, resume-supported stages:

```
experiments/experiment7b/
├── config.py                           # Central configuration & hyperparameters
├── phase0_audit.py                     # Environment, dependency, and artifact check
├── phase1_candidate_selection.py       # Stratified selection of 18 candidate pairs
├── phase2_actual_merges.py             # Stage A: In-place parameter merges & D_actual
├── phase3_individual_interventions.py  # Stage B1: Individual expert ablations D(i)
├── phase4_joint_interventions.py       # Stage B2: Joint expert ablations D(i, j) & I(i, j)
├── phase5_analysis.py                  # Primary/Secondary analysis, plots, Gate 1/2
├── run_all.py                          # 1-Click end-to-end orchestrator
└── utils/
    ├── model_utils.py                  # Loading, clean restoration, expert patching
    ├── evaluation.py                   # Wikitext KL & ARC capability pipelines
    ├── feature_loader.py               # Feature extraction & CARE descriptor calculations
    └── statistics.py                   # Spearman rho, bootstrap CIs, calibration
```

---

## 📊 Candidate Pair Selection (Phase 1)

18 expert pairs sampled from central Layer 8 ($\binom{64}{2} = 2,016$ total pairs) using fixed deterministic seed `SEED = 42`:
- **Group A (Safe / High-Ranking Candidates):** Top 15% lowest $D_{pred}$ (6 pairs).
- **Group B (Moderate / Mid-Ranking Candidates):** Percentiles 40%–60% of $D_{pred}$ (6 pairs).
- **Group C (Unsafe / Low-Ranking Candidates):** Bottom 15% highest $D_{pred}$ (6 pairs).

---

## ⚡ Execution Instructions

### 1. Local Machine (Development & Precomputation)
The local machine runs lightweight audits, candidate selection, and post-experiment statistical analysis:
```bash
# 1. Run Pre-Execution Audit
python experiments/experiment7b/phase0_audit.py

# 2. Run Candidate Selection
python experiments/experiment7b/phase1_candidate_selection.py
```

### 2. GPU VM (Model Merges & Interventions)
On the GPU VM with PyTorch and CUDA:
```bash
# Set device and batch size if needed
export CARE_MOE_GPU_ID=0
export CARE_MOE_BATCH_SIZE=4

# Run all remaining phases or full orchestrator
python experiments/experiment7b/run_all.py
```
Or run each phase individually:
```bash
python experiments/experiment7b/phase2_actual_merges.py
python experiments/experiment7b/phase3_individual_interventions.py
python experiments/experiment7b/phase4_joint_interventions.py
python experiments/experiment7b/phase5_analysis.py
```

---

## 📁 Results Directory (`results/exp7b/`)

All outputs are saved to `results/exp7b/`:
- `candidates/candidate_pairs.csv`: The 18 selected candidate pairs with metadata.
- `merges/actual_merge_results.csv`: Post-merge measurements ($D_{actual}^{KL}$, ARC metrics).
- `interventions/individual_interventions.csv`: Individual expert ablations ($D(i)$).
- `interventions/joint_interventions.csv`: Joint ablations ($D(i, j)$) and interactions ($I(i, j)$).
- `analysis/EXPERIMENT_7B_FINAL_REPORT.md`: Comprehensive final scientific report.
- `analysis/statistical_summary.json`: Machine-readable correlation values, CIs, and Gate decisions.
- `plots/`: Publication-ready scatter plots and degeneracy landscape visualizations.
