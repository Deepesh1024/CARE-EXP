# Table 2: Static vs Adaptive Ablation

| Target Experts | Static PPL | Adaptive PPL | Static KL | Adaptive KL | KL Reduction |
|----------------|------------|--------------|-----------|-------------|--------------|
| 60 | 12.112 | **11.548** | 2.477 | **2.036** | ~17.8% |
| 56 | 14.820 | **13.812** | 6.820 | **5.143** | ~24.6% |
| 48 | 21.032 | **20.136** | 17.694 | **14.881** | ~15.9% |

*Note: Both methods start from identical initial states and use identical merge operators.*
