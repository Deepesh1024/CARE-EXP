# Phase A Profiling Report

Based on the profiling of Phase 3 execution, here is the breakdown of runtime per component:

## Runtime Breakdown
- **SMACOF Optimization** (`n=51` training experts, `q=4`, `max_iter=300`, `n_init=2`): **~1.87 seconds**
- **Out-of-Sample (OOS) Embedding** (L-BFGS-B, 13 test experts, 2 restarts): **~0.015 seconds total**
- **OOS per test expert**: **~0.0012 seconds**

### Extrapolated Runtime per Fold (for a single Q value)
- **Time per (Fold, Q)**: ~1.88 seconds
- The OOS optimization is surprisingly fast and completely negligible compared to the SMACOF MDS training.

### Caching Analysis & Redundancy Check
**Question**: Are we rerunning SMACOF or regenerating null matrices unnecessarily?
**Answer**: **No.** 
- `Z_train` depends on the specific distance matrix `D`, the specific subset of training experts `train_idx`, and the dimensionality `q`. 
- Because we use `KFold(shuffle=True)` with different random seeds for each repetition, every single `(repetition, fold)` combination produces a unique `train_idx` split.
- Null matrices are generated exactly once per realization and passed into the cross-validation loop.
- Therefore, there are **zero identical `(D, train_idx, q)` conditions** across the entire pipeline. Caching `Z_train` would yield zero cache hits because we never evaluate the exact same training split and distance matrix twice.

The sheer volume of necessary, independent conditions is what dominates the execution time:
`Total SMACOF Fits = (N_LAYERS) × (N_Q_VALUES) × (N_FOLDS) × [ (N_ORACLE_REPS) + 2 × (N_NULL_REALIZATIONS) ]`

## Pilot Configuration
I have updated `config.py` to run a significantly reduced pilot version of the experiment to quickly validate the shape of the results:
- `Q_VALUES` = [2, 4, 6, 8]
- `N_REPETITIONS` = 2 (for Oracle)
- `N_NULL_REALIZATIONS` = 1
- `SMACOF_MAX_ITER` = 300
- `SMACOF_N_INIT` = 2
- `OOS_N_RESTARTS` = 2

**Estimated Pilot Runtime**:
- Total SMACOF Fits = 3 layers × 4 q-values × 5 folds × (2 Oracle + 1 NullA + 1 NullB) = 240 fits.
- At ~1.87s per fit, the entire Phase 3 pilot should take **under 8 minutes** to complete.

I am now proceeding to run the pilot pipeline.
