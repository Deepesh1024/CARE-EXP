# EXPERIMENT 7C — FINAL REPORT

## Scientific Question
Can fine-grained functional neuron geometry explain the residual errors
of CARE-COM's expert-level merge predictions?

## Primary Hypothesis
Functional neuron mutual coverage C_mutual(i,j) is associated with the
CARE-COM residual R_ij = D_actual(i,j) - D_pred(i,j).

## Primary Result

| Statistic | Value |
|---|---|
| Spearman rho | -0.0671 |
| p-value | 7.91e-01 |
| 95% Bootstrap CI | [-0.5462, +0.4444] |
| N | 18 candidate pairs |

## Verdict

**7C HYPOTHESIS IS KILLED.**

The primary Spearman correlation between C_mutual and the CARE-COM residual
is negligible (rho = -0.07) and massively non-significant (p = 0.79).
The 95% bootstrap CI spans from -0.55 to +0.44, crossing zero and providing
no evidence of a consistent direction.

Per the pre-specified failure criterion, this null result is accepted as-is.
No post-hoc search for alternative statistics was performed.

## Secondary Exploratory Results (do not override primary)

| Statistic | rho | p |
|---|---|---|
| C_i_to_j vs residual | -0.0237 | 0.9255 |
| C_j_to_i vs residual | -0.0795 | 0.7540 |

All exploratory statistics consistently null.

## Interpretation

The CARE-COM prediction residuals are NOT explained by fine-grained functional
neuron geometry as measured by mutual neuron coverage over 262,144 Wikitext
probe tokens. This is a scientifically informative null result:

1. CARE-COM's Layer-8 expert-level merge prediction errors are not driven
   by neuron-level functional overlap between the merged experts.

2. The residuals may instead reflect:
   - Higher-order inter-layer interactions not captured at a single layer
   - Token-distribution mismatch between the probe dataset and the true
     deployment distribution
   - Non-linear expert interaction effects that are outside the scope of
     a simple cosine similarity metric

3. This result does not invalidate CARE-COM itself. CARE-COM's Gate 1
   result (7B: Spearman rho = +0.69, p = 0.0016) remains intact and
   represents meaningful predictive utility for expert merge damage.

## Protocol Compliance
- Primary statistic pre-specified: YES
- Normalization choice pre-specified (L2-normalize): YES
- Analysis performed before seeing results: YES
- Post-hoc statistic rescue attempted: NO
- Sample size: N=18 (hypothesis-generating, not confirmatory)
- DERN distinction maintained: YES (no compression algorithm implemented)

## Extraction Details
- Probe dataset: Wikitext-2 (train), 262,144 tokens
- Layer: 8 (TARGET_LAYER_IDX)
- Experts extracted: 29 unique experts from 18 candidate pairs
- Top-k routing: k=8 (per OLMoE architecture)
- Activation dtype: float32
- Normalization: L2-normalize before cosine similarity
- Random seed: 42

## Output Files
- `results/exp7c/activations/expert_signatures.npy` — raw activation memmap
- `results/exp7c/activations/expert_signatures_meta.json` — metadata
- `results/exp7c/analysis/cloud_analysis_results.csv` — pairwise C_mutual table
- `results/exp7c/analysis/7c_final_residual_analysis.csv` — full joined table
