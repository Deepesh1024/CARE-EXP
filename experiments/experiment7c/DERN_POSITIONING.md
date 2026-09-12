# DERN Positioning Note: CARE-EXP 7C vs DERN

## The DERN Conceptual Overlap
DERN (Dynamic Expert Recombination Network) demonstrates that fine-grained, neuron-level structure and compatibility can be leveraged for Mixture-of-Experts (MoE) compression, expert recombination, and routing. Both DERN and the analyses in CARE-EXP 7C evaluate cross-expert "neuron-level geometry" and "functional signatures."

## CARE 7C's Scope and Limitation
**CARE 7C is NOT claiming to introduce neuron-level recombination.**

Unlike DERN, Experiment 7C does not perform:
- Neuron recombination
- Neuron transplantation
- Neuron-level expert reconstruction
- Neuron matching used directly as a compression algorithm
- A DERN-like expert decomposition/compression method

## CARE 7C as a Diagnostic
CARE 7C is solely a **diagnostic experiment**. Its only purpose is to take the residual errors from an *already-existing expert-level CARE-COM merge predictor* and test whether those residual errors can be statistically explained by fine-grained functional neuron geometry.

The pipeline is strictly:
1. Observe CARE-COM expert-level prediction.
2. Observe actual merge outcome (from 7B).
3. Compute the prediction residual (Actual - Predicted).
4. Measure functional neuron geometry (Mutual Coverage).
5. Test whether the neuron geometry statistically explains the residual.

**Conclusion**: Do not make unsupported novelty claims in 7C regarding MoE compression. The novelty of 7C lies in explaining the failure modes (residuals) of expert-level combinatorial prediction (CARE-COM) through the lens of internal functional geometry.
