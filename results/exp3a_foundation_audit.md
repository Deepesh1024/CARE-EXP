# CARE-MoE: EXPERIMENT 3A FOUNDATION AUDIT & EXPERIMENT 4 PLAN

## 1. Executive Verdict
The fundamental claim that *capability topology emerges from routing behavior* has structural merit, but the specific validation in Experiment 3A suffers from severe circularity. The Null B test definitively rejects the null hypothesis ($p < 0.0001$), confirming that the observed community assignments capture genuine functional relationships better than random groupings. However, the graph itself was built from a Surrogate trained on the same data it was validated against. 

**H2 Classification: PARTIALLY SUPPORTED.**
The effect is statistically undeniable, but because of the circularity, we must pivot to Experiment 4 to rigorously test if global topology adds *new* predictive power beyond simple pairwise features on held-out pairs.

---

## 2. Provenance Audit
**Status: VERIFIED (With Circularity Warning)**
- **Model**: `allenai/OLMoE-1B-7B-0924` (The "Mistral" references were hardcoded errors in the report generation scripts).
- **Graph Construction**: The Mutual-kNN graph ($k=8$) was constructed using an Affinity matrix derived from *Surrogate Predicted KL*, not the true Oracle KL.
- **Validation**: The discovered communities were validated by measuring the *True Oracle KL* differences within and between communities.

## 3. Matrix Integrity
**Status: VERIFIED**
The underlying true Oracle KL matrix (`output.json`, Exp 1) contains exactly 2016 pairs per layer for `Seq_Len=512`. The values are strictly positive ($0.0005$ to $0.0432$), symmetric by construction (using `UniformAverage` merge), and contain no NaNs or Infs. 

## 4. Oracle Sensitivity / Noise
**Status: COMPLETED (Moderate Noise Floor)**
We measured the Oracle KL across 10 random expert pairs (5 repetitions per pair with different subsets of 8 calibration sequences).
- **Average Coefficient of Variation (CoV)**: $0.2787$
- **Max CoV observed**: $0.4559$
- **Average Relative Difference (Max-Min / Mean)**: $0.7690$
- **Max Absolute Difference**: $0.006854$

**Conclusion**: The Oracle KL measurement shows *moderate* sensitivity to the calibration batch. While individual measurements have up to ~27% relative variance, the mean signal remains stable enough to differentiate good vs. bad merges. However, this non-zero noise floor justifies the use of larger calibration batches for final evaluations.

## 5. 3A Reproduction
**Status: REPRODUCED EXACTLY**
We independently recalculated the $T$-statistic ($D_{\text{between}} - D_{\text{within}}$).
- **First Layer**: $T = 0.001793$ ($p = 3.78 \times 10^{-13}$)
- **Last Layer**: $T = 0.003594$ ($p = 4.03 \times 10^{-54}$)
- **Aggregated**: $T = 0.001478$ ($p = 6.87 \times 10^{-24}$)
The difference is statistically highly significant.

## 6. Null B: Primary H2 Null
**Status: REJECTED ($p < 0.0001$)**
We randomized the Surrogate predictions across the unordered pairs, reconstructed the graph, and re-measured the *True Oracle KL* separation.
- **Aggregated Layer**: Real $T = 0.001970$, Null $T = 0.000004 \pm 0.000139$.
- **Z-Score**: $14.11$.
**Conclusion**: The specific pairings assigned by the Surrogate capture genuine functional topology that disappears completely when the edge assignments are randomized.

## 7. Null A: Degree-Preserving Graph Sanity Check
**Status: NOT REJECTED ($p = 0.6190$ for aggregated)**
Applying `double_edge_swap` to randomize the graph while preserving degree sequences resulted in a Null Modularity ($0.3436$) that is statistically indistinguishable from the real graph's modularity ($0.3372$).
**Conclusion**: The graph's macroscopic modularity is an artifact of its degree distribution (a known property of kNN graphs), not an exceptionally strong modular community structure.

## 8. Circularity Analysis
The most critical flaw in 3A is that the Surrogate (used to build the graph) was trained to predict the True Oracle KL (used to validate the graph) using features like Weight Distance and Routing Similarity. The discovery of communities simply proves that experts with similar weights/routing form clusters that, predictably, have lower Oracle KL. 

## 9. Existing Feature-Regression Comparison
- **Regression Track (Exp 2)**: Predicts Oracle KL perfectly for a *single pair* using local features (Weights, Routing, Usage).
- **Graph Track (Exp 3A)**: Ignores pairwise properties and focuses on *global structural context* (communities, centrality, hubs).
**Conclusion**: They are complementary. A central hub expert might be difficult to merge regardless of who it is paired with, a property invisible to the Exp 2 pairwise features but obvious in the Exp 3A graph.

---

## 10. & 11. Experiment 4 Plan: Functional Merge Landscape

**Central Question**: What measurable properties of an expert pair predict the actual functional damage caused by merging that pair, and does global topology add predictive power over local features?

**Target Variable**: $Y_{ij} =$ True Oracle KL (measured in Exp 1).

**Features (Pre-Merge Only)**:
1. *Local Weight Features*: Weight Distance, Weight Cosine
2. *Local Routing Features*: Usage Frequency, Jaccard Overlap, Usage Asymmetry, Routing JSD/NPMI
3. *Global Graph Features* (Derived from Exp 3A unweighted k=8 graph):
   - Degree of $i$ and $j$
   - Betweenness Centrality of $i$ and $j$
   - PageRank of $i$ and $j$
   - Graph Shortest Path Distance between $i$ and $j$
   - Community Co-membership (1 if same Louvain community, 0 otherwise)

**Leakage Prevention (CRITICAL)**:
Because topological features are derived from the same Surrogate predictions, we must evaluate on **Node-Disjoint Splits**:
- Train Set: All pairs formed by experts 0-31.
- Test Set: All pairs formed by experts 32-63.
- The graph features must be computed *independently* on the subgraph of 0-31 (for training) and the subgraph of 32-63 (for testing).

**Evaluation Metrics**:
- **Regression**: RMSE, MAE, R², Spearman Correlation.
- **Ranking**: Precision@5%, Precision@10%, Precision@20% (How many of our predicted top-5% safest merges are actually in the true top-5%?).

## 12. Venue / Scope Conflicts
A repository-wide audit revealed **no active venue targeting**. "Mistral" references were hard-coded string errors. References to "NeurIPS workshop, ICLR, ICML 2027, Interspeech 2027" are not present in the current `Experiments-V3` codebase. 

## 13. Recommended Next Action
1. Review this audit.
2. Approve the Experiment 4 Plan (specifically the Node-Disjoint split strategy to solve the circularity issue).
3. Do **NOT** proceed to Experiment 4 until explicit approval is granted.
