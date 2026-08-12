# EXPERIMENT 3A: NULL A AUDIT (GRAPH SANITY CHECK)

> **WARNING:** This is ONLY a structural sanity check. It determines if the observed graph has more community structure than a random graph with the same degree sequence. **It does NOT prove that the graph or its communities are functionally meaningful.**

## Methodology
We took the $k=8$ Mutual-kNN unweighted graph and applied `nx.double_edge_swap` 500 times per realization to randomize topology while strictly preserving the degree of every node.
We ran Louvain community detection on 1000 such null graphs and computed the modularity.

## Results

### Layer: First
- **Real Modularity**: 0.3006
- **Null Modularity Mean (Std)**: 0.2740 (0.0222)
- **Z-Score**: 1.20
- **Empirical p-value**: 0.1180

### Layer: Middle
- **Real Modularity**: 0.3722
- **Null Modularity Mean (Std)**: 0.3712 (0.0174)
- **Z-Score**: 0.06
- **Empirical p-value**: 0.4640

### Layer: Last
- **Real Modularity**: 0.2943
- **Null Modularity Mean (Std)**: 0.2664 (0.0202)
- **Z-Score**: 1.38
- **Empirical p-value**: 0.0900

### Layer: Aggregated
- **Real Modularity**: 0.3372
- **Null Modularity Mean (Std)**: 0.3436 (0.0196)
- **Z-Score**: -0.33
- **Empirical p-value**: 0.6190

## Conclusion
If $p < 0.05$, the real graph exhibits significantly stronger community structure than degree-preserving random graphs. This validates the graph construction pipeline but not the underlying functional claim.