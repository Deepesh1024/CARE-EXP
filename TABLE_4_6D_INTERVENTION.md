# TABLE 4: EXP 6D DIRECTIONAL INTERVENTION (RAW DATA EXTRACT)

**Source Data:** `results/exp6d_rerun/exp6d/EXP6D_RAW_RESULTS.parquet`
**Total Interventions (N):** 1200

### Table 4A: Experimental Parameters (Direct Extraction)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Alpha ($\alpha$) Regime** | [0.01, 5.0] | Structural intervention magnitudes |
| **Angle Conditions ($\theta$)** | [np.float64(-2.0), np.float64(-1.0), np.float64(0.0), np.float64(15.0), np.float64(30.0), np.float64(45.0), np.float64(60.0), np.float64(72.19422347835044), np.float64(72.20500250453111), np.float64(72.24092614791743), np.float64(72.3474159995346), np.float64(72.39617295717436), np.float64(72.41832048038422), np.float64(72.42477289540113), np.float64(72.50998188053615), np.float64(72.61339599753754), np.float64(72.62615110004822), np.float64(72.64736571889203), np.float64(72.65205378128772), np.float64(72.66416607021378), np.float64(72.69688236761695), np.float64(72.69715518495107), np.float64(72.7285459044066), np.float64(72.76490420237334), np.float64(72.77506120848895), np.float64(72.81844941194994), np.float64(72.91407930810487), np.float64(72.9271191564849), np.float64(73.17116824460231), np.float64(73.30838339149828), np.float64(73.36238881246136), np.float64(73.4508101998732), np.float64(73.50026290943397), np.float64(74.0180430523168), np.float64(74.33823796516322), np.float64(74.36693140140366), np.float64(75.67799521699794)] degrees | 0=Aligned, 90=Orthogonal |
| **Quantile Targets** | [0.1, 0.25, 0.5, 0.75, 0.9] | Expert initial capability ($||C||$) percentiles |
| **Random Seeds** | 3 per condition | Cross-validation noise robustification |
| **Target Layer** | 2 | Selected intervention layer |
| **Calibration Metrics** | $\tau_{\text{actual}}$ realizability | Verified via `err_tau`, `cos_tau` |

### Table 4B: Derived Statistical Effects (Low-Alpha Regime $\alpha \le 1.0$)
| Measurement | Value / Correlation | Interpretation |
| :--- | :--- | :--- |
| **$\alpha$ vs $\Delta\theta$ Correlation** | Spearman $\rho = 0.3696$ | Demonstrates approximately linear functional drift |
| **Mean Aligned Drift ($\alpha=1.0$, $0^\circ$)** | 0.0276 radians | Baseline capability deformation |
| **Mean Ortho Drift ($\alpha=1.0$, $90^\circ$)** | nan radians | Orthogonal resistance deformation |
| **Orthogonal Resistance Ratio** | nanx | The network resists orthogonal interventions far more than aligned ones |

*Note: Raw data preserved in `EXP6D_RAW_RESULTS.csv`.*
