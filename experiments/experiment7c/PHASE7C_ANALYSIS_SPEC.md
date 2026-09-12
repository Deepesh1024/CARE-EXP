# PHASE 7C ANALYSIS SPECIFICATION

## 1. Scientific Question
Can fine-grained functional neuron geometry explain the residual errors of CARE-COM's expert-level merge predictions?

## 2. Primary Hypothesis
Functional neuron mutual coverage is associated with the CARE-COM residual:
$C_{mutual}(i,j) \leftrightarrow R_{ij}$

## 3. Exact Residual Definition
For each candidate pair $(i,j)$ from the 18 candidate pairs tested in Experiment 7B:
$R_{ij} = D_{actual}(i,j) - D_{pred}(i,j)$
Where:
- $D_{actual}(i,j)$ is the actual measured merge damage (KL divergence vs base model) from 7B Phase 2 (`Error_KL`).
- $D_{pred}(i,j)$ is the CARE-COM predicted merge damage from 7B.

## 4. Exact Neuron Signature Definition
For neuron $k$ in expert $i$, the functional signature is the sequence of post-activation scalar values over a deterministic set of probe inputs $x_1, ..., x_M$:
$\phi_{i,k} = [a_{i,k}(x_1), ..., a_{i,k}(x_M)]$
where the SAME probe inputs are used for every candidate expert. Layer 8 experts are evaluated. The signature uses exactly the 1024-dimensional hidden states directly following the activation function but before the down-projection.

## 5. Exact Similarity Definition
For expert pair $(i,j)$, the similarity between neuron $k$ in expert $i$ and neuron $l$ in expert $j$ is defined by cosine similarity:
$S_{kl} = cosine(\phi_{i,k}, \phi_{j,l})$

## 6. Exact Mutual Coverage Definition
The overall functional mutual coverage for pair $(i,j)$ is defined as the mean maximum similarity across neurons in both directions:
$C_{i \to j} = \frac{1}{1024} \sum_{k=1}^{1024} \max_{l} S_{kl}$
$C_{j \to i} = \frac{1}{1024} \sum_{l=1}^{1024} \max_{k} S_{kl}$
$C_{mutual} = 0.5 \times (C_{i \to j} + C_{j \to i})$

## 7. Primary Statistical Test
The primary test is the Spearman rank correlation between $C_{mutual}$ and $R_{ij}$ across the candidate pairs.

## 8. Sample Size
$N = 18$ expert pairs. Due to the small sample size, this experiment is explicitly classified as **hypothesis-generating**, not definitive confirmation. Do not claim a general law from N=18.

## 9. Probe Dataset/Protocol
The probe dataset is the EXACT common Wikitext probe protocol used for the 7B Wikitext KL evaluation. 
- A fully deterministic sequence of tokens from Wikitext-2.
- `RANDOM_SEED = 42`.
- Exact token ordering and sequence lengths must be preserved across experts.

## 10. Normalization Protocol
**Choice: B (Simply L2-normalized)**
Rationale: The post-activation neuron outputs are strictly non-negative (assuming ReLU/SiLU/GeLU like activations where most relevant values are positive, or in general, their absolute magnitudes and zero-points matter). Mean-centering (Choice A) before cosine similarity converts the metric to Pearson correlation, which destroys the distinction between a neuron that never fires and one that fires actively, as well as shifts the zero-point of the activation geometry. L2-normalization preserves the directional geometry of the non-negative activation space, which is critical for representing "functional presence" over the sequence. We will use L2-normalization strictly.
*Note: Any alternatives evaluated must be strictly marked as exploratory.*

## 11. Exclusion Criteria
Only the exact 18 candidate pairs from Experiment 7B (comprising ~36 unique experts) are included. The pairs must match the identities exactly. If any pair fails to extract, the entire correlation must be reconsidered. No pairs will be dropped post-hoc.

## 12. Secondary Exploratory Analyses
Secondary exploratory statistics may include:
- Unidirectional coverage: $C_{i \to j}$ or $C_{j \to i}$
- Mean pairwise similarity
- Threshold coverage (e.g., proportion of neurons with $\max_l S_{kl} > 0.8$)
- Down-projection weight norm scaling ($q_{i,k} = \|W_{down}[:,k]\|_2$) applied to similarity.
*All above must be clearly labeled as EXPLORATORY and cannot replace the primary statistic.*

## 13. DERN Distinction
DERN utilizes conceptual overlap with neuron-level cross-expert compatibility for actual expert recombination/compression. **CARE 7C is NOT claiming to introduce neuron-level recombination.** 7C is solely a diagnostic experiment to test whether fine-grained functional neuron geometry explains residual errors in an already-existing expert-level CARE-COM predictor. 7C does not implement a new compression method or alter CARE-COM.

## 14. Interpretation Rules
- The cloud represents functional response similarity.
- It does NOT automatically measure: importance, causal contribution, capability, or output contribution. No such claims will be made.

## 15. Failure Criteria
If the primary relationship (Spearman rho between $C_{mutual}$ and $R$) is weak, non-significant, or inconsistent, the experiment has failed to support the hypothesis. Do NOT search for another statistic simply to rescue 7C. Accept the null result.
