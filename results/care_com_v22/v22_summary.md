# CARE-COM v2.2 Controlled Adaptive Compression Validation

Random Seeds Evaluated: 3

## Perplexity Results
| Method                | PPL@64          | PPL@60           | PPL@56           |
|:----------------------|:----------------|:-----------------|:-----------------|
| Adaptive              | 6210377.5       | 5071247.5        | 4925146.0        |
| Static                | 6210377.5       | 5229712.0        | 5562377.0        |
| Random (mean 3 seeds) | 8.3289 ± 0.0000 | 10.4748 ± 0.1041 | 13.5614 ± 0.3647 |

## Marginal Damage Results
| Method                | Mean Marginal KL    | Cumulative KL       |
|:----------------------|:--------------------|:--------------------|
| Adaptive              | 0.03848738550799113 | 0.307899084063929   |
| Static                | 0.05177062420982243 | 0.41416499367857945 |
| Random (mean 3 seeds) | 0.0709 ± 0.0042     | 0.5671 ± 0.0337     |

## Pair Selection Divergence
|   Step | Static Pair   | Adaptive Pair   |   Static Capability Dist |   Adaptive Capability Dist |   Static Marginal KL |   Adaptive Marginal KL |   Delta KL (Static - Adaptive) | Note                                             |
|-------:|:--------------|:----------------|-------------------------:|---------------------------:|---------------------:|-----------------------:|-------------------------------:|:-------------------------------------------------|
|     64 | (8, 13)       | (13, 15)        |                0.0156636 |                  0.0157447 |           0.0742783  |             0.04669    |                    0.0275882   | Direct comparison at identical state M_t         |
|     63 | (8, 14)       | (12, 13)        |                0.0158342 |                  0.0176232 |           0.0276849  |             0.0283461  |                   -0.000661146 | State diverged (KL measured at different states) |
|     62 | (7, 10)       | (10, 12)        |                0.018043  |                  0.0206349 |           0.0634922  |             0.0325904  |                    0.0309018   | State diverged (KL measured at different states) |
|     61 | (0, 11)       | (7, 10)         |                0.0199426 |                  0.0174652 |           0.061997   |             0.0403483  |                    0.0216487   | State diverged (KL measured at different states) |
|     60 | (0, 8)        | (5, 12)         |                0.0212271 |                  0.0231192 |           0.00556301 |             0.0567924  |                   -0.0512294   | State diverged (KL measured at different states) |
|     59 | (7, 10)       | (1, 4)          |                0.0239374 |                  0.0233013 |           0.0659687  |             0.0546187  |                    0.01135     | State diverged (KL measured at different states) |
|     58 | (1, 4)        | (4, 15)         |                0.0242498 |                  0.0225851 |           0.0579099  |             0.0432676  |                    0.0146423   | State diverged (KL measured at different states) |
|     57 | (0, 3)        | (4, 6)          |                0.0257172 |                  0.0219642 |           0.057271   |             0.00524553 |                    0.0520255   | State diverged (KL measured at different states) |

