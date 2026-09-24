# Negative Results (Appendix)

1. **Taylor Neuron Sensitivity (Exp 7A/7A.51):**
   - Hypothesis: Weighting expert similarity by gradient-based neuron importance would improve correlation with damage.
   - Result: Correlation dropped substantially relative to simple activation tracking. 
   - Conclusion: First-order gradients were too noisy over the calibration set.

2. **Interaction Mechanism (Exp 7B):**
   - Hypothesis: Explicitly modeling the co-activation of experts would yield better pairs.
   - Result: High computational cost with no statistically significant improvement in Precision@K.

3. **Neuron-Cloud/Mutual Coverage (Exp 7C):**
   - Hypothesis: Measuring set-intersection of active neurons rather than Euclidean capability distance.
   - Result: Failed to capture the magnitude of output deviations.

These negative results motivate why CARE-COM uses the simplest, most robust capability representation (Euclidean distance on normalized expected activations) combined with exact KL validation.
