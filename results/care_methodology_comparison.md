# CARE Methodology Comparison: Feature-Regression vs Graphs

## Overview
The CARE framework currently explores functional routing topology through two distinct analytical lenses:
1. **The Feature-Regression Track** (Experiments 1, 1.5, 2): Focuses on *predicting* the pairwise Oracle KL divergence (merge damage) using a cheap proxy features available at inference time.
2. **The Graph/Topology Track** (Experiment 3A): Focuses on *discovering communities* of redundant experts by treating predicted functional distances as a graph.

This document compares these methodologies, their features, and highlights the dependencies (and circularities) between them.

---

## The Feature-Regression Track (Exp 2)

**Goal:** Predict the actual Oracle KL for a pair $(i, j)$ without running a forward pass.
**Target:** $Y_{ij} =$ True Oracle KL divergence.
**Features Available at Inference Time:**
- **Weight-Space Features:** `Weight_Distance`, `Weight_Cosine`
- **Activation/Output Space Features:** `Activation_Similarity`, `Output_Similarity`
- **Routing/Usage Features:** `Routing_Similarity`, `Usage_Frequency`, `Jaccard_Overlap`, `Usage_Asymmetry`, `Routing_JSD_Proxy`, `Routing_NPMI_Proxy`, `Specialization_Diff`

**Methodology:**
- A supervised XGBoost model is trained on a subset of experts to predict the target $Y_{ij}$ using the features above.
- The model acts as a "Surrogate" Oracle, estimating the functional damage of merging any pair of experts based purely on their structural and routing similarities.

---

## The Graph/Topology Track (Exp 3A)

**Goal:** Identify macroscopic clusters ("communities") of redundant experts that can be merged together safely.
**Target:** Not applicable (Unsupervised clustering).

**Methodology:**
- It uses the frozen XGBoost Surrogate from Exp 2 to predict the Oracle KL for all $\binom{64}{2}$ expert pairs.
- It transforms these predictions into an affinity matrix, builds a Mutual-kNN graph, and runs Louvain community detection.
- **Validation:** The discovered communities are then validated by comparing the *true* Oracle KL of within-community merges vs. between-community merges.

---

## Comparison and Dependency (Circularity) Warning

The two approaches are not independent. Experiment 3A is entirely dependent on the predictions of Experiment 2.

### The Circularity Issue
The graph constructed in Experiment 3A does *not* represent the true functional topology of the model. It represents the topology of the *XGBoost surrogate's feature space*.
- Because the XGBoost model heavily weights features like `Weight_Distance` and `Routing_Similarity`, the resulting graph communities are essentially clusters of experts that have similar weights and routing patterns.
- When Experiment 3A validates these communities using the true Oracle KL, it is effectively just proving that "experts with similar weights and routing patterns have lower Oracle KL when merged."
- This was already proven by the Feature-Regression track (which showed these features are predictive of Oracle KL).

### Synergies and Next Steps (Experiment 4)
Rather than competing, the two approaches should be combined.
The structural graph features (e.g., centrality, betweenness, community co-membership) capture *global* topological relationships that the local pairwise features in Exp 2 miss.
- **Experiment 4** will integrate these tracks by treating the graph properties derived in 3A as *new features* for a predictive regression model. This will test whether global topological position provides additional predictive power for actual merge damage beyond simple pairwise similarities.
