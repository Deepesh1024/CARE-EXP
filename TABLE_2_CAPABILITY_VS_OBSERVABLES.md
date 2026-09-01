# TABLE 2: CAPABILITY VS EXISTING OBSERVABLES

**Source References:**
- `results/exp1/report.md` (Experiment 1)
- `results/exp2/report2.md` (Experiment 2)

### Table 2A: Univariate Correlation with Capability Drift (Oracle KL)

*Extracted from Exp 2 (Out-of-Distribution Test Splits, N=1,488)*

| Feature / Observable | Category | Spearman $\rho$ | Pearson $r$ | VIF | 
| :--- | :--- | :--- | :--- | :--- |
| **Usage Frequency** | Routing / Utilization | +0.5573 | +0.4006 | 1.218 |
| **Output Similarity** | Structural | +0.3429 | +0.0538 | 1.354 |
| **Usage Asymmetry** | CARE Descriptor | +0.2011 | +0.2773 | N/A |
| **Jaccard Overlap** | Routing / Utilization | +0.2041 | +0.1020 | 1.812 |
| **Weight Cosine** | Weight Similarity | +0.0964 | +0.0299 | 1.944 |
| **Activation Similarity** | Activation Similarity | -0.0053 | -0.0120 | 1.320 |
| **Weight Distance** | Weight Similarity | -0.0104 | -0.0014 | 1.115 |
| **Routing Similarity** | Routing / Utilization | -0.1049 | -0.0565 | 1.642 |

### Table 2B: Known Methodological Contradictions

*Note: Per instructions, these values are extracted exactly as they appear in the source reports without attempting silent reconciliation.*

| Parameter | Value in Exp 1 (`report.md`) | Value in Exp 2 (`report2.md`) | Impact / Note |
| :--- | :--- | :--- | :--- |
| **Calibration Seq Length** | Scaled across $N \in \{64, 128, 256, 512\}$ | Fixed at `Seq_Len = 256` | The 512-token correlations in Exp 1 may slightly mismatch the canonical values reported in Exp 2. |
| **Layer Aggregation** | Segmented evaluation (first, middle, last) | Global training with `Relative_Depth` interaction | Exp 1 qualitative scatter plots mask the global metrics computed in Exp 2. |
| **Target Representation** | Raw Oracle KL Divergence | Normalized Oracle KL | Exp 2 explicitly scales the features and targets using `RobustScaler` fit on train. |
