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

## 4. Activation Extraction (Phase 6)
- **Dataset**: `Salesforce/wikitext-2-raw-v1`
- **Subset**: `train`
- **Quantity**: 262,144 tokens (`NUM_TOKENS_TO_EXTRACT`)
- **Tokenization**: `max_length=1024`, `truncation=True` (Same exact sequence order and structure as 7B phase 2).
- **Target Experts**: The precise set of unique Layer-8 experts involved in the 18 selected candidate pairs from 7B.
- **Probe Point**: Post-activation neuron state, mathematically `act_fn(gate_proj(x)) * up_proj(x)`. This represents the expert's outgoing functional signal before being down-projected back into the model's residual stream.

## 5. Activation Filtering (Revised)
- `attention_mask` is explicitly tracked and applied.
- Padded token positions are discarded from calculations.
- For a given pair of experts $(i, j)$, neuron response signatures are restricted exclusively to the mutually activated token positions $T_{ij}$:
  $$T_{ij} = \{t : r_i(t) > 0 \text{ and } r_j(t) > 0\}$$

## 6. Cloud Similarity Calculation (Phase 7)
For a pair of experts $(i, j)$ and their mutually activated tokens $T_{ij}$:
1. Let $\mathbf{A}^{(i)}$ be the $[1024 \times T_{ij}]$ post-activation tensor for expert $i$.
2. Normalize vectors: L2 normalization is applied to each of the 1024 neuron vectors over the $T_{ij}$ tokens (Protocol B: Scale Invariance).
3. Compute the $[1024 \times 1024]$ cross-expert cosine similarity matrix $\mathbf{S}$.
4. Compute the nearest-neighbor directed coverage:
   $$C_{i \to j} = \frac{1}{1024} \sum_{k=1}^{1024} \max_{l} \mathbf{S}_{k,l}$$
   $$C_{j \to i} = \frac{1}{1024} \sum_{l=1}^{1024} \max_{k} \mathbf{S}_{k,l}$$
5. The **Mutual Coverage ($C_{mutual}$)** is defined as:
   $$C_{mutual} = 0.5 \times (C_{i \to j} + C_{j \to i})$$

## 7. Primary Statistical Test (N=18)
The primary test is the Spearman rank correlation between $C_{mutual}$ and $R_{ij}$ across ALL 18 candidate pairs. The primary analysis MUST retain all 18 pairs to respect the pre-specified protocol.

## 8. Sensitivity Statistical Test (N=14)
A pre-specified sensitivity analysis will be run strictly on the 14 pairs satisfying $T_{ij} \ge 50$. This is a practical minimum-support threshold motivated by the instability of estimating 1024-dimensional nearest-neighbor cosine similarity from extremely small numbers of observations.

## 9. Sample Size
$N = 18$ expert pairs (Primary). $N = 14$ pairs (Sensitivity). Due to the small sample size, this experiment is explicitly classified as **hypothesis-generating**, not definitive confirmation. Do not claim a general law from N=18.

## 10. Probe Dataset/Protocol
The probe dataset is the EXACT common Wikitext probe protocol used for the 7B Wikitext KL evaluation. 
- A fully deterministic sequence of tokens from Wikitext-2.
- `RANDOM_SEED = 42`.
- Exact token ordering and sequence lengths must be preserved across experts.

## 11. Normalization Protocol
**Choice: B (Simply L2-normalized)**
Rationale: The post-activation neuron outputs are strictly non-negative (assuming ReLU/SiLU/GeLU like activations where most relevant values are positive, or in general, their absolute magnitudes and zero-points matter). Mean-centering (Choice A) before cosine similarity converts the metric to Pearson correlation, which destroys the distinction between a neuron that never fires and one that fires actively, as well as shifts the zero-point of the activation geometry. L2-normalization preserves the directional geometry of the non-negative activation space, which is critical for representing "functional presence" over the sequence. We will use L2-normalization strictly.
*Note: Any alternatives evaluated must be strictly marked as exploratory.*

## 12. Exclusion Criteria
Only the exact 18 candidate pairs from Experiment 7B (comprising ~36 unique experts) are included. The pairs must match the identities exactly. If any pair fails to extract, the entire correlation must be reconsidered. No pairs will be dropped post-hoc.

## 13. Secondary Exploratory Analyses
Secondary exploratory statistics may include:
- Unidirectional coverage: $C_{i \to j}$ or $C_{j \to i}$
- Mean pairwise similarity
- Threshold coverage (e.g., proportion of neurons with $\max_l S_{kl} > 0.8$)
- Down-projection weight norm scaling ($q_{i,k} = \|W_{down}[:,k]\|_2$) applied to similarity.
*All above must be clearly labeled as EXPLORATORY and cannot replace the primary statistic.*

## 14. DERN Distinction
DERN utilizes conceptual overlap with neuron-level cross-expert compatibility for actual expert recombination/compression. **CARE 7C is NOT claiming to introduce neuron-level recombination.** 7C is solely a diagnostic experiment to test whether fine-grained functional neuron geometry explains residual errors in an already-existing expert-level CARE-COM predictor. 7C does not implement a new compression method or alter CARE-COM.

## 15. Interpretation Rules
- The cloud represents functional response similarity.
- It does NOT automatically measure: importance, causal contribution, capability, or output contribution. No such claims will be made.

## 16. Failure Criteria
If the primary relationship (Spearman rho between $C_{mutual}$ and $R$) is weak, non-significant, or inconsistent, the experiment has failed to support the hypothesis. Do NOT search for another statistic simply to rescue 7C. Accept the null result.
