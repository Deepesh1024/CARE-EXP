# Experiment 9: Robust Baseline and CARE Incremental Value
**Status:** PROTOCOL FROZEN (Pre-Execution)

## 1. Primary Scientific Question
Does CARE geometry provide incremental predictive information about parameter-merge KL divergence beyond what is captured by Usage mass and RW-L2 functional distance?

## 2. Experimental Controls
To ensure absolute comparability with Experiment 4 and prevent data leakage:
- **Target Intervention:** Oracle Parameter-Merge KL Divergence (from Exp 3b, evaluated on Split B).
- **Pair Set:** The exact same 2,016 expert pairs from the middle layer (Layer 8).
- **Cross-Validation:** The exact same 5-fold expert-disjoint partitions used in Experiment 4 (`cv_splits.json`). 
- **Predictor Data Source:** All predictors (Usage, RW-L2, Local 11, CARE Geometry) are computed strictly from pre-merge representations on Split A. The target KL is computed strictly on post-merge models on Split B.

## 3. Predictor Definitions
1. **Local-11:** The original 11 local features used in Experiment 4 Model A.
2. **Usage (Symmetric):** The independent routing masses $u_i$ and $u_j$, combined as a symmetric sum ($u_i + u_j$).
3. **RW-L2:** The dynamic, routing-weighted L2 functional distance of expert outputs ($||e_i(x) - e_j(x)||_2^2$ weighted by router probability).
4. **CARE Geometry:** The latent metric distance $||z_i - z_j||_2$ derived from the original Exp 4 MDS embedding. (No modifications to CARE).

## 4. Models Evaluated
All learned models utilize the same XGBoost hyperparameters defined in Experiment 4.

- **Model 0 (Reference):** Local-11 (Original Exp 4 Model A)
- **Model 1 (New Baseline):** Usage + RW-L2
- **Model 2 (Geometry Only):** CARE Geometry (Direct correlation, no training)
- **Model 3 (Incremental 1):** CARE + Usage
- **Model 4 (Incremental 2):** CARE + Usage + RW-L2

## 5. Metrics and Reporting
- **Primary Metric:** Expert-disjoint Spearman rank correlation ($\rho$) between model predictions (or direct distances) and the Oracle Parameter-Merge KL.
- **Secondary Metric:** Expert-disjoint Out-of-Sample $R^2$.

**Statistical Rigor:**
- Results will be reported both pooled across all predictions and fold-wise.
- Confidence Intervals will be computed using clustered bootstrap over source expert $j$.
- Incremental value will be assessed via direct paired statistical comparisons of fold-wise performance (e.g., comparing Model 1 vs Model 4 fold-by-fold).

## 6. Execution Constraints
- No GWS, Hessians, or additional nonlinear feature engineering will be introduced.
- No modifications to the CARE embedding algorithm.
- No leakage between Split A (predictors) and Split B (target).
