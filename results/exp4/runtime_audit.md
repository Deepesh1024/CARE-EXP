# Experiment 4 Runtime & Reproducibility Audit

## Overview
This audit verifies the runtime characteristics, model independence, and correct execution of CARE-MoE Experiment 4, without altering the methodology or re-running the full experiment.

## 1. Wall-Clock Time
Based on real-time execution sampling (P0F0) and file-system `mtime` deltas:
- **MDS (SMACOF)**: ~0.03s
- **OOS Embedding**: ~0.13s
- **XGBoost (A & C)**: ~0.41s
- **Total Fold Time**: ~0.6s – 1.0s

## 2. SMACOF Iterations
For every restart, `sklearn.manifold.MDS` converges rapidly:
- Restart 1: 16 iterations
- Restart 2: 11 iterations
- Restart 3: 20 iterations
- Restart 4: 19 iterations
- Restart 5: 19 iterations

## 3. MDS Stress
Final stress converges tightly for every fit:
- Stress values: `0.000488`, `0.000510`, `0.000500`, `0.000495`, `0.000494`.
*(Note: Initial stress is not explicitly exposed by sklearn, but final stress is highly consistent).*

## 4. `max_iter=3000` Active
**Confirmed:** `config.py` correctly passes `SMACOF_MAX_ITER = 3000` to the `MDS` constructor. However, it is never reached because early stopping is triggered by the `eps=1e-4` tolerance.

## 5. All 5 MDS Restarts Execute
**Confirmed:** `SMACOF_N_INIT = 5` is strictly enforced. The audit script confirms 5 separate fits execute for every training sequence.

## 6. No Cached Embeddings/Predictions Reused
**Confirmed:** Inside `run_all.py`, the `run_fold` function recalculates `Z_train`, `Z_test`, `geom_test`, `geom_train`, `pred_a`, `pred_b`, and `pred_c` directly from data. It does not load `.npy` files unless skipping a completed fold via `--resume`.

## 7. OOS Coordinates Independently Optimized
**Confirmed:** `mds_embedding.py` contains a strict `for t, test_exp in enumerate(test_idx):` loop that uses `L-BFGS-B` to optimize each test expert's coordinate using *only* distances to training experts.

## 8. Model A Retrained Every Fold
**Confirmed:** `train_model_a(X_train, y_train)` is explicitly called at step 3 of `run_fold`. XGBoost models are initialized from scratch.

## 9. Model C Retrained Every Fold
**Confirmed:** `train_model_c(X_train, y_train, geom_train)` is explicitly called at step 5 of `run_fold`.

## 10. COMPLETE Markers
**Confirmed:** `write_fold_complete()` is invoked as the very last instruction of `run_fold`, only after all artifact `atomic_write_json` and `_npy_atomic_save` calls complete successfully.

## 11. Discrepancy Explanation
**Discrepancy:** The reported operation times (assumed to be long for matrix embedding/gradient boosting) vs. actual wall-clock fold time.
**Explanation:** 
1. **Microscopic Operation Time:** MDS on a ~42x42 matrix converges in under 20 iterations (<0.05s). Total fold computation is extremely lightweight, completing in roughly 1 second.
2. **Timestamp Artifacts:** Any large gaps or out-of-order `COMPLETE` timestamps (e.g., `P0F0` completing *after* `P1` folds, or 20-minute gaps) are not due to computation time. They are deliberately induced by the Phase 3 Resume Test (which deletes and re-runs `P0F0`) and standard manual pausing/resuming of the orchestration script.
