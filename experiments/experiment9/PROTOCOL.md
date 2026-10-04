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
3. **RW-L2:** The dynamic, routing-weighted L2 functional distance of expert outputs.
   **Exact Formula:** $D_{\mathrm{RW-L2}}(i,j) = \mathbb{E}_{x} \left[ P(j|x) \left\|e_i(x)-e_j(x)\right\|_2^2 \right]$
   *Note on weighting:* The expectation is over **all valid tokens** in the dataset, and $P(j|x)$ is the raw softmax gating probability of the source expert $j$ prior to top-K masking. This perfectly matches the Exp 8 control.
4. **CARE Geometry:** The latent metric distance $||z_i - z_j||_2$ derived from the original Exp 4 MDS embedding. (No modifications to CARE).

## 4. Models Evaluated
All learned models utilize the same XGBoost hyperparameters defined in Experiment 4.

- **Model 0 (Reference):** Local-11 (Original Exp 4 Model A)
- **Model 1 (New Baseline):** Usage + RW-L2
- **Model 2 (Geometry Only):** CARE Geometry (Direct correlation, no training)
- **Model 3 (Incremental 1):** CARE + Usage
- **Model 4 (Incremental 2):** CARE + functional controls (CARE + Usage + RW-L2)

## 5. Metrics and Reporting
- **Primary Metric:** Expert-disjoint Spearman rank correlation ($\rho$) between model predictions (or direct distances) and the Oracle Parameter-Merge KL.
- **Secondary Metric:** Expert-disjoint Out-of-Sample $R^2$.

**Statistical Rigor:**
- Results will be reported both pooled across all predictions and fold-wise.
- Confidence Intervals will be computed using clustered bootstrap over source expert $j$.
- Incremental value will be assessed via direct paired statistical comparisons of fold-wise performance (e.g., comparing Model 1 vs Model 4 fold-by-fold).

## 6. Predefined Outcomes

### Outcome A — Strong CARE increment
$\rho(\text{CARE+controls}) \gg \rho(\text{Usage+RW-L2})$
*Conclusion:* CARE geometry contains incremental relational information beyond exposure and routed functional displacement.

### Outcome B — CARE adds little
$\rho(\text{CARE+controls}) \approx \rho(\text{controls})$
*Conclusion:* Much of the apparent predictive advantage of CARE can be explained by simpler functional/exposure statistics.

### Outcome C — CARE geometry itself loses to RW-L2
$\rho(\text{CARE Geometry Only}) < \rho(\text{RW-L2})$
*Conclusion:* The current CARE representation does not outperform a direct routed functional-distance statistic for this parameter-merge target.

### Outcome D — CARE helps specifically at selective K
*Conclusion:* If CARE retains an advantage specifically in the low-damage candidate-selection regime (e.g., Precision@10 or 25), that is highly relevant to CARE-COM, even if global $\rho$ does not dominate.

## 7. Execution Constraints
- No GWS, Hessians, or additional nonlinear feature engineering will be introduced.
- No modifications to the CARE embedding algorithm.
- No leakage between Split A (predictors) and Split B (target).
