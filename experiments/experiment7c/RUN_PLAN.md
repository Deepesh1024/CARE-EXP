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
## Status: READY
The pipeline is fully configured. It is updated to compute a routing-conditioned mutual coverage mechanism, filtering padded tokens and only analyzing mutually activated positions ($T_{ij} \ge 50$ for sensitivity).

## Phase 6: Extraction
**Status: COMPLETED (using original extraction). No new inference required.**
We preserve the original 29GB memmap data, which contains activations for all tokens.

## Phase 7: Cloud Similarity (Revised)
**Status: READY**
Script: `experiments/experiment7c/phase7_cloud_analysis.py`
Steps:
1. Reconstruct exact global `attention_mask`.
2. Apply mask to discard padding.
3. Restrict each candidate pair to its mutually activated tokens $T_{ij}$.
4. Compute normalized cosine similarity and mutual coverage.
5. Export to `results/exp7c/revised/analysis/cloud_analysis_results.csv`.

## Phase 8: Residual Join
**Status: READY**
Script: `experiments/experiment7c/phase8_residual_analysis.py`
Steps:
1. Join revised cloud analysis with 7B residual errors by `pair_id`.
2. Compute `residual = D_actual_KL - D_pred`.
3. Export to `results/exp7c/revised/analysis/7c_final_residual_analysis.csv`.

## Report Generation
Script: `experiments/experiment7c/generate_revised_report.py`
Steps:
1. Load Phase 8 results.
2. Compute primary N=18 Spearman correlation.
3. Compute sensitivity N=14 Spearman correlation ($T_{ij} \ge 50$).
4. Check Pair 7-32 diagnostic influence.
5. Print final scientific interpretation report.
- [ ] Output the final joined analysis table.

## PHASE F: Primary Spearman analysis
- [ ] Execute primary Spearman rank correlation between $C_{mutual}$ and the prediction residual.
- [ ] Output rho, p-value, and confidence interval.

## PHASE G: Exploratory diagnostics
- [ ] Execute any predefined secondary/exploratory analyses (unidirectional coverage, etc.).
- [ ] Document clearly as exploratory.

## PHASE H: Independent replication
- [ ] Execute independent replication **if and only if** the primary result in Phase F is promising. Do not jump directly to this phase.
