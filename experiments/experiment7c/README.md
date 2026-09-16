# Experiment 7C: Functional Neuron Geometry vs. CARE-COM Residuals

This directory contains the protocol and code for CARE-EXP Experiment 7C.

## Background
Experiment 7B demonstrated that CARE-COM has meaningful predictive utility for expert merge damage, but its joint-interaction hypothesis (Gate 2) failed to explain the remaining prediction errors.

Experiment 7C tests whether fine-grained functional neuron geometry (i.e. neuron-level mutual coverage over an activation dataset) explains the residual errors of CARE-COM's expert-level merge predictions.

- **Phase 6**: Extract layer 8 expert neuron responses over 262,144 Wikitext tokens.
- **Phase 7**: Calculate cross-expert neuron functional similarity, restricted to tokens where BOTH experts are activated ($T_{ij} = \{t : r_i(t) > 0 \land r_j(t) > 0\}$). Computes mutual coverage ($C_{mutual}$).
- **Phase 8**: Analyze association between $C_{mutual}$ and the merge residual $R_{ij} = D_{actual} - D_{pred}$.

*Note: The 7C mechanism uses a routing-conditioned, mutually-activated-token neuron cloud. Original unconditioned results are preserved in `results/exp7c/original/`.*

## Status: PREPARATION PHASE
The scripts and analysis specifications in this directory are prepared for execution.
Full activation extraction (~19 GB, 262K tokens) requires execution on a GPU VM.

## Contents
- `PHASE7C_ANALYSIS_SPEC.md`: The rigorous scientific data contract and hypothesis specification.
- `RUN_PLAN.md`: The phase-by-phase execution plan.
- `DERN_POSITIONING.md`: Note clarifying the distinction between 7C (a diagnostic experiment) and DERN (a compression algorithm).
- `phase6_extract_activations.py`: Streams Layer 8 expert post-activation representations to a memmap.
- `phase7_cloud_analysis.py`: Computes pairwise functional neuron cosine similarities.
- `phase8_residual_analysis.py`: Joins neuron cloud similarity with 7B merge prediction residuals.
