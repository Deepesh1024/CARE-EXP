# Experiment 7C: Functional Neuron Geometry vs. CARE-COM Residuals

This directory contains the protocol and code for CARE-EXP Experiment 7C.

## Background
Experiment 7B demonstrated that CARE-COM has meaningful predictive utility for expert merge damage, but its joint-interaction hypothesis (Gate 2) failed to explain the remaining prediction errors.

Experiment 7C tests whether fine-grained functional neuron geometry (i.e. neuron-level mutual coverage over an activation dataset) explains the residual errors of CARE-COM's expert-level merge predictions.

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
