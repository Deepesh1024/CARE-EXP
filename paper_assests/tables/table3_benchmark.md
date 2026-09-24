# Table 3: External Benchmark Comparison (WikiText-2 Perplexity)

| Method | 60 Experts | 56 Experts | 48 Experts |
|--------|------------|------------|------------|
| CARE-Adaptive (Ours) | **11.548** | 13.812 | 20.136 |
| CARE-Static | 12.112 | 14.820 | 21.032 |
| Sub-MoE | 11.811 | **13.735** | **19.477** |
| HC-SMoE | ~12.303 | 14.795 | 30.069 |
| M-SMoE | ~12.303 | 15.283 | 27.655 |
| Random | ~12.758 | 17.298 | 38.104 ± ~13.5 |

*Note: Lower is better. External baselines lack verified tracking of Cumulative KL.*
