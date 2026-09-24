# Experiment 4 Metric Definition: Precision@K

## Mathematical Definition
In Experiment 4, **Precision@K** measures the overlap between the top-K pairs selected by a predictive model and the true top-K pairs as determined by the Oracle KL divergence.

Specifically:
- **K** refers to a fixed budget of top candidate pairs (e.g., K=10, 25, 50) rather than a percentage or quantile.
- **Positive/Correct Condition:** A predicted pair is considered "correct" if it exists in the actual top-K set of pairs with the lowest Oracle KL divergence. Lower KL is considered positive (less damage).
- **Threshold:** The threshold is implicitly defined by the K-th best Oracle KL value within the evaluation set. It is not fixed apriori, but dynamically determined by the empirical damage landscape of the specific fold.
- **Sorting Direction:** Candidates are ranked in ascending order for all methods (Local Baseline, Capability Geometry, Combined Model), meaning a lower score corresponds to a prediction of lower functional damage.
- **Evaluation Scope:** Precision@K is computed **within each cross-validation fold (per layer)** and then averaged across all folds. It is not computed globally across all layers at once, which ensures that candidates are only competing with other candidates from the same model partition.
- **Ties:** In the event of score ties, standard stable sorting applies, though ties at exactly the K-th boundary are exceedingly rare for floating-point distances.

Precision@K is formulated as:
`Precision@K = | Predicted_Top_K ∩ Oracle_Top_K | / K`
