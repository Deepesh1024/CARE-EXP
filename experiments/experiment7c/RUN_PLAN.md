# Experiment 7C: RUN PLAN

This plan dictates the exact progression for executing Experiment 7C. Adhere to it strictly.

## PHASE A: Repository/protocol audit
- [x] Inspect 7B directories and existing utilities.
- [x] Identify location of 7B final prediction and actual merge results.
- [x] Establish data contract and experiment layout.

## PHASE B: 512-token extraction sanity test
- [ ] Run `phase6_extract_activations.py` locally with `NUM_TOKENS_TO_EXTRACT = 512`.
- [ ] Validate shape, expert IDs, data types, and absence of degenerate values.

## PHASE C: Full activation extraction on GPU VM
- [ ] Set `NUM_TOKENS_TO_EXTRACT = 262144`.
- [ ] Execute `python experiments/experiment7c/phase6_extract_activations.py` on the GPU VM.
- [ ] Ensure sufficient disk space (~19 GB) for the memmap array.

## PHASE D: Pairwise neuron-cloud construction
- [ ] Run `python experiments/experiment7c/phase7_cloud_analysis.py`.
- [ ] Process only the 18 specific candidate pairs.
- [ ] Save the compact similarity result table.

## PHASE E: Join with 7B residuals
- [ ] Run `python experiments/experiment7c/phase8_residual_analysis.py`.
- [ ] Join the results of Phase D with the `results/exp7b/merges/actual_merge_results.csv` and CARE-COM predictions.
- [ ] Output the final joined analysis table.

## PHASE F: Primary Spearman analysis
- [ ] Execute primary Spearman rank correlation between $C_{mutual}$ and the prediction residual.
- [ ] Output rho, p-value, and confidence interval.

## PHASE G: Exploratory diagnostics
- [ ] Execute any predefined secondary/exploratory analyses (unidirectional coverage, etc.).
- [ ] Document clearly as exploratory.

## PHASE H: Independent replication
- [ ] Execute independent replication **if and only if** the primary result in Phase F is promising. Do not jump directly to this phase.
