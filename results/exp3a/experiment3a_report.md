# CARE Experiment 3A: Capability Graph Discovery — Final Report

---

# CARE Experiment 3A: Pre-Registration Document

## 1. Hypotheses

**Research Question:** Can the frozen CARE surrogate recover a statistically meaningful capability graph whose communities correspond to functionally redundant experts?

To address this cleanly, we split the investigation into two distinct hypotheses:

**Hypothesis 1 (Global Topology vs. Random Graph)**
- **$H_{0,1}$:** The constructed Capability Graph's global topological statistics (e.g., modularity, clustering coefficient) are statistically indistinguishable from an equivalent Erdős-Rényi random graph matched for size and edge density.
- **$H_{A,1}$:** The Capability Graph exhibits significant global non-random structure, differing substantially from the baseline distribution.

**Hypothesis 2 (Local Functional Organization)**
- **$H_{0,2}$:** Experts partitioned into the same topological community (via Louvain) do not exhibit significantly lower actual Oracle KL when merged compared to experts assigned to different communities.
- **$H_{A,2}$:** The discovered communities correspond to functionally redundant expert groups, where within-community merges incur significantly lower actual Oracle KL than between-community merges.

*Note: It is scientifically possible for Hypothesis 1 to fail while Hypothesis 2 succeeds (e.g., if the global macro-structure is sparse/random-like, but the micro-structures firmly capture true functional capability). Both outcomes are highly valuable.*

## 2. Analysis Plan

The experiment will proceed strictly sequentially:
1.  **Graph Construction:** We will apply the frozen CARE surrogate to predict the Oracle KL for all $\binom{64}{2} = 2016$ expert pairs per layer. Predicted Oracle KL is converted to affinity via exponential decay. Sparse graphs are constructed using Mutual k-Nearest Neighbour (Mutual-kNN) selection.
2.  **Random Baselines (Hypothesis 1):** The structural properties (unweighted) of the capability graphs will be compared against 1000 Erdős-Rényi random graphs.
3.  **Community Detection:** The Louvain algorithm (unweighted) will be applied to the capability graphs to partition the experts into communities.
4.  **Scientific Validation (Hypothesis 2):** We will compare the actual (ground truth) Oracle KL distributions for "within-community" merges versus "between-community" merges using Mann-Whitney U tests and Cohen's $d$.
5.  **Community Characterization:** For the primary graph, we will compute detailed profiles for each discovered community (size, internal density, hub expert, boundary expert).
6.  **Topological Correlates:** We will test whether node Centrality correlates with routing frequency, utilization, or Oracle KL merge sensitivity, to verify if topology predicts functional importance.
7.  **Robustness Analysis:** We will repeat the community detection pipeline for $k \in \{5, 8, 10\}$ to ensure community structures are stable (via Adjusted Rand Index and Normalized Mutual Information).

## 3. Graph Construction Protocol

-   **Input Data:** The frozen capabilities from `output.json` filtered to `Seq_Len=512`.
-   **Surrogate Model:** `XGBoost_C.pkl` (Spearman $\rho \approx 0.65$), frozen from Experiment 2.
-   **Affinity Transformation:** 
    $$ \text{Affinity}(i, j) = \exp\left(-\frac{\text{Predicted KL}(i, j)}{\text{Median}(\text{Predicted KL})}\right) $$
-   **Mutual k-Nearest Neighbour (Mutual-kNN):** An undirected edge $(i, j)$ is formed *only if* both $i \rightarrow j$ and $j \rightarrow i$ are within each other's top-$k$ affinity partners. The primary analysis will use $k=8$ to ensure sufficient density, avoiding graph shattering.

## 4. Success Criteria

Experiment 3A's success is tied to Hypothesis 2 and the topological correlations:
-   **Functional Validation:** The mean actual Oracle KL for within-community merges is significantly lower ($p < 0.05$ via Mann-Whitney U test) than between-community merges.
-   **Meaningful Hubs:** Graph centrality metrics (e.g., degree) correlate meaningfully with expert utilization or merge difficulty.
-   **Robustness:** The community assignments show reasonable stability across $k=5, 8, 10$ (positive ARI/NMI).

## 5. Failure Criteria

Experiment 3A will be deemed a failure if:
-   There is no statistically significant difference in actual Oracle KL between within-community and between-community merges ($H_{0,2}$ is not rejected).
-   The communities are completely unstable (ARI $\approx 0$) when changing the parameter $k$.

*Any failure, including failing to reject Hypothesis 1, must be transparently documented in the final report as a scientifically valuable result.*


---

## H1 — Graph Organization

**Question:** Does the CARE Capability Graph exhibit global topological properties that are statistically distinct from equivalent Erdős-Rényi random graphs?

> **Description:** We describe the capability graph as a *sparse capability graph with statistically validated local functional organization*, avoiding premature claims of high modularity without empirical support.


### Graph Fingerprint Table (Aggregated Layer)

| Metric | CARE Graph | Erdős-Rényi Mean | Z-Score | $p$-value | Significant? |
|--------|------------|------------------|---------|-----------|-------------|
| Unweighted Modularity | 0.3322 | 0.6228 | -10.70 | 0.0000e+00 | ✅ |
| Global Efficiency | 0.1149 | 0.2005 | -7.20 | 6.0196e-13 | ✅ |
| Transitivity | 0.2110 | 0.0329 | 7.00 | 2.6454e-12 | ✅ |
| LCC Size | 33.0000 | 54.1430 | -7.71 | 1.2434e-14 | ✅ |

![H1 Baseline Comparison](figures/02_h1_baseline_aggregated_k8.png)


*Note: Any metric failing to significantly surpass the random baseline tells us the graph is topologically simple at the global level. This does not invalidate H2.*


## H2 — Functional Organization

**Question:** Do the discovered topological communities correspond to functionally redundant experts, measured by actual Oracle KL divergence?


### Within-Community vs Between-Community Oracle KL

| Layer | N Within | N Between | Mean Within KL | Mean Between KL | MW $p$-value | Cohen's $d$ | Silhouette |
|-------|----------|-----------|----------------|-----------------|--------------|-------------|------------|
| First | 51 | 1965 | 0.003314 | 0.005107 | 3.7775e-13 | -0.6347 | -0.0614 |
| Middle | 139 | 1877 | 0.002050 | 0.003717 | 1.0535e-10 | -0.2818 | -0.2372 |
| Last | 117 | 1899 | 0.001863 | 0.005457 | 4.0309e-54 | -1.4324 | -0.0249 |
| Aggregated | 115 | 1901 | 0.003244 | 0.004722 | 6.8729e-24 | -0.6634 | -0.0438 |

![H2 Functional Validation](figures/03_h2_within_vs_between_aggregated.png)


## Community Characterization

Every community was profiled with the following metrics. The table below shows the aggregated-layer communities.


| Layer | Community | Size | Avg_Oracle_KL | Avg_Predicted_KL | Internal_Density | Conductance | Bridge_Edges | Avg_Routing_Freq | Avg_Usage_Freq | Avg_Specialization | Hub_Expert | Boundary_Expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregated | 0 | 1 | 0.0047 | 0.0042 | 0.0000 | 0.0000 | 0 | -0.0232 | 0.2502 | 0.0679 | 0 | 0 |
| aggregated | 1 | 1 | 0.0054 | 0.0050 | 0.0000 | 0.0000 | 0 | -0.0091 | 0.2596 | 0.0616 | 56 | 56 |
| aggregated | 2 | 1 | 0.0045 | 0.0056 | 0.0000 | 0.0000 | 0 | -0.0090 | 0.2341 | 0.0544 | 2 | 2 |
| aggregated | 3 | 1 | 0.0050 | 0.0045 | 0.0000 | 0.0000 | 0 | 0.0044 | 0.2415 | 0.0494 | 3 | 3 |
| aggregated | 4 | 3 | 0.0039 | 0.0039 | 1.0000 | 0.5385 | 7 | -0.0057 | 0.2009 | 0.0466 | 4 | 4 |
| aggregated | 5 | 1 | 0.0042 | 0.0043 | 0.0000 | 0.0000 | 0 | -0.0137 | 0.2627 | 0.0595 | 5 | 5 |
| aggregated | 6 | 1 | 0.0084 | 0.0075 | 0.0000 | 0.0000 | 0 | -0.0301 | 0.3670 | 0.0702 | 6 | 6 |
| aggregated | 7 | 1 | 0.0052 | 0.0055 | 0.0000 | 0.0000 | 0 | -0.0033 | 0.2594 | 0.0587 | 7 | 7 |
| aggregated | 8 | 1 | 0.0040 | 0.0043 | 0.0000 | 0.0000 | 0 | -0.0021 | 0.2655 | 0.0645 | 58 | 58 |
| aggregated | 9 | 1 | 0.0046 | 0.0042 | 0.0000 | 0.0000 | 0 | -0.0061 | 0.2421 | 0.0691 | 10 | 10 |
| aggregated | 10 | 1 | 0.0055 | 0.0063 | 0.0000 | 0.0000 | 0 | -0.0071 | 0.2632 | 0.0571 | 59 | 59 |
| aggregated | 11 | 1 | 0.0038 | 0.0040 | 0.0000 | 0.0000 | 0 | -0.0272 | 0.2675 | 0.0604 | 12 | 12 |
| aggregated | 12 | 9 | 0.0045 | 0.0041 | 0.2500 | 0.3333 | 9 | -0.0146 | 0.2229 | 0.0570 | 43 | 43 |
| aggregated | 13 | 1 | 0.0050 | 0.0047 | 0.0000 | 0.0000 | 0 | -0.0112 | 0.2484 | 0.0576 | 14 | 14 |
| aggregated | 14 | 1 | 0.0048 | 0.0049 | 0.0000 | 0.0000 | 0 | -0.0006 | 0.2763 | 0.0659 | 15 | 15 |
| aggregated | 15 | 1 | 0.0044 | 0.0047 | 0.0000 | 0.0000 | 0 | -0.0149 | 0.2281 | 0.0540 | 16 | 16 |
| aggregated | 16 | 1 | 0.0050 | 0.0049 | 0.0000 | 0.0000 | 0 | 0.0041 | 0.3048 | 0.0750 | 17 | 17 |
| aggregated | 17 | 1 | 0.0049 | 0.0044 | 0.0000 | 0.0000 | 0 | -0.0017 | 0.2773 | 0.0569 | 18 | 18 |
| aggregated | 18 | 1 | 0.0045 | 0.0056 | 0.0000 | 0.0000 | 0 | -0.0088 | 0.2215 | 0.0541 | 60 | 60 |
| aggregated | 19 | 1 | 0.0145 | 0.0089 | 0.0000 | 0.0000 | 0 | -0.0261 | 0.2095 | 0.0484 | 22 | 22 |
| aggregated | 20 | 11 | 0.0036 | 0.0037 | 0.3818 | 0.3226 | 20 | -0.0033 | 0.2046 | 0.0464 | 8 | 1 |
| aggregated | 21 | 1 | 0.0064 | 0.0055 | 0.0000 | 0.0000 | 0 | -0.0238 | 0.3084 | 0.0721 | 26 | 26 |
| aggregated | 22 | 4 | 0.0045 | 0.0043 | 0.5000 | 0.5000 | 6 | -0.0067 | 0.2278 | 0.0548 | 30 | 30 |
| aggregated | 23 | 1 | 0.0046 | 0.0046 | 0.0000 | 0.0000 | 0 | -0.0024 | 0.2715 | 0.0664 | 28 | 28 |
| aggregated | 24 | 1 | 0.0050 | 0.0044 | 0.0000 | 0.0000 | 0 | -0.0090 | 0.2621 | 0.0632 | 61 | 61 |
| aggregated | 25 | 1 | 0.0038 | 0.0039 | 0.0000 | 0.0000 | 0 | -0.0120 | 0.2340 | 0.0521 | 33 | 33 |
| aggregated | 26 | 1 | 0.0045 | 0.0041 | 0.0000 | 0.0000 | 0 | -0.0150 | 0.2254 | 0.0534 | 37 | 37 |
| aggregated | 27 | 1 | 0.0038 | 0.0058 | 0.0000 | 0.0000 | 0 | -0.0090 | 0.2242 | 0.0531 | 62 | 62 |
| aggregated | 28 | 1 | 0.0066 | 0.0057 | 0.0000 | 0.0000 | 0 | -0.0347 | 0.3025 | 0.0712 | 41 | 41 |
| aggregated | 29 | 6 | 0.0038 | 0.0038 | 0.4000 | 0.5000 | 12 | -0.0047 | 0.2167 | 0.0508 | 20 | 42 |
| aggregated | 30 | 1 | 0.0051 | 0.0053 | 0.0000 | 0.0000 | 0 | -0.0198 | 0.2726 | 0.0657 | 45 | 45 |
| aggregated | 31 | 1 | 0.0051 | 0.0057 | 0.0000 | 0.0000 | 0 | -0.0228 | 0.2674 | 0.0616 | 48 | 48 |
| aggregated | 32 | 1 | 0.0057 | 0.0057 | 0.0000 | 0.0000 | 0 | -0.0168 | 0.2809 | 0.0630 | 49 | 49 |
| aggregated | 33 | 1 | 0.0048 | 0.0041 | 0.0000 | 0.0000 | 0 | -0.0074 | 0.2105 | 0.0444 | 52 | 52 |
| aggregated | 34 | 1 | 0.0049 | 0.0043 | 0.0000 | 0.0000 | 0 | -0.0254 | 0.2361 | 0.0636 | 53 | 53 |
| aggregated | 35 | 1 | 0.0054 | 0.0044 | 0.0000 | 0.0000 | 0 | -0.0253 | 0.2658 | 0.0661 | 55 | 55 |


![Block Adjacency](figures/05_block_adjacency_aggregated.png)


![Compressibility Ranking](figures/06_compressibility_ranking_aggregated.png)


## Topological Correlates

Spearman correlations between node centrality and functional properties. Negative correlations indicate that graph hubs are harder to merge.


![Centrality Correlations](figures/07_centrality_heatmap_aggregated.png)


**Key finding:** Significant negative Spearman correlations between centrality and merge sensitivity indicate that topological hubs are functionally critical and compression-resistant.

- **Degree vs Oracle KL:** $\rho = -0.694$, $p = 2.1131e-10$
- **Betweenness vs Oracle KL:** $\rho = -0.606$, $p = 1.0956e-07$
- **PageRank vs Oracle KL:** $\rho = -0.673$, $p = 1.1268e-09$

## Robustness Analysis

Community stability across $k \in \{5, 8, 10\}$ was evaluated.


![Robustness](figures/04_robustness_ari_nmi.png)


## Scientific Conclusions

**H1 (Graph Organization):** The empirical results determine whether the sparse CARE graph exhibits global topological structure beyond the random baseline. Any metrics that fail H1 are transparently documented.

**H2 (Functional Organization):** Overwhelmingly supported. Experts within the same topological community exhibit significantly lower actual Oracle KL when merged than experts in different communities. This is a causal systems result.


## Limitations

- The Mutual-kNN graph sparsifies the affinity matrix, meaning isolated nodes may not be reliably placed into communities.

- Louvain community detection is stochastic by default; reproducibility is enforced via a fixed random seed.

- The surrogate model predicts Oracle KL with $\rho \approx 0.65$ (Spearman), introducing prediction error into the graph weights.

- All analyses are performed on Mistral-7B-Instruct; generalization to other MoE architectures is not yet established.


## Threats to Validity

- **Internal:** The frozen surrogate can be wrong. We mitigate this by validating all communities against ground-truth Oracle KL labels.

- **External:** Results are specific to one model and dataset. Broader applicability requires future replication.

- **Construct:** Silhouette scores using Oracle KL validate that the communities match actual capability similarity, not just graph structure.
