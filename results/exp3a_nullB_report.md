# EXPERIMENT 3A: NULL B AUDIT (EDGE PERMUTATION)

## Hypothesis H2
> *The assignment of functional divergence values to specific expert pairs contains non-random community structure.*

## Methodology
We randomized the Predicted KL values across the unordered expert pairs 1000 times.
For each null realization, we reconstructed the Affinity matrix, Mutual-kNN graph ($k=8$), and Louvain communities.
We then computed the exact same statistic $T = D_{between} - D_{within}$ using the *true* Oracle KL.

## Results

### Layer: First
- **Real $T$**: 0.001835
- **Null $T$ (95% CI)**: [-0.000342, 0.000363]
- **Z-Score**: 10.32
- **Empirical p-value**: 0.0000

### Layer: Middle
- **Real $T$**: 0.001719
- **Null $T$ (95% CI)**: [-0.000660, 0.000685]
- **Z-Score**: 4.91
- **Empirical p-value**: 0.0000

### Layer: Last
- **Real $T$**: 0.003522
- **Null $T$ (95% CI)**: [-0.000330, 0.000318]
- **Z-Score**: 20.86
- **Empirical p-value**: 0.0000

### Layer: Aggregated
- **Real $T$**: 0.001970
- **Null $T$ (95% CI)**: [-0.000283, 0.000275]
- **Z-Score**: 14.11
- **Empirical p-value**: 0.0000

## Conclusion
If $p < 0.05$ across layers, Null B is rejected, confirming that the specific pairing of experts contains non-random functional structure beyond just the density distribution of distances.