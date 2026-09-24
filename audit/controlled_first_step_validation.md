# CONTROLLED_FIRST_STEP_COMPARISON

## Validation of the Static vs Adaptive First-Step Candidates

We independently verify the comparison between the first-step selected candidate of Static CARE-COM and Adaptive CARE-COM on the 64-expert `allenai/OLMoE-1B-7B-0924` model:

- **Static candidate:** `[8, 13]` | Capability distance ≈ `0.0156` | Actual Oracle KL ≈ `0.074`
- **Adaptive candidate:** `[13, 15]` | Capability distance ≈ `0.0157` | Actual Oracle KL ≈ `0.046`

### Independent Verification Check

1. **Both pairs were valid in the same initial model:** Yes. Since this occurs at Step 1 (64 experts → 63 experts), both candidates refer to the original expert indices in the exact same `M_64` model checkpoint.
2. **Capability distances were computed using the same capability state:** Yes. Both Static and Adaptive CARE-COM use the identical capability distance matrix for the very first step because the model has not yet been modified.
3. **KL was measured using exactly the same calibration inputs:** Yes. Both candidates were evaluated by the oracle using the identical 128-sequence WikiText-2 calibration set.
4. **Both physical merges used the same merge operator:** Yes. Both employed the identical physical parameter merge (`linear_combine_experts`) which performs a weighted average of the expert weights based on gating activation frequencies.
5. **No post-selection information leaked into candidate generation:** Yes. The capability geometry was purely observational, based on the pre-merge forward pass activations.
6. **The difference is not caused by pair reindexing:** Yes. At Step 1, no prior deletions have occurred, meaning index `8`, `13`, and `15` map identically to the original pretrained experts.

### Conclusion

The comparison is mathematically and methodologically valid. It robustly supports the statement:
*"Capability proximity does not completely determine intervention consequence."*

However, as per the audit constraints, this single example should **not** be used to claim that capability distance is a poor predictor overall. The broader pair-level analysis (Experiment 4) establishes that capability distance *is* a strong predictor (rho ≈ 0.75), but contains variance that necessitates exact functional evaluation.
