# EXP3C DATA AUDIT

## 1. Available Checkpoints
dict_keys(['checkpoint_10', 'checkpoint_40', 'checkpoint_70', 'checkpoint_100'])

## 2. Exact Position
| checkpoint_id   |   training_step |   pct |
|:----------------|----------------:|------:|
| checkpoint_10   |          120000 |   9.8 |
| checkpoint_40   |          490000 |  40.2 |
| checkpoint_70   |          795000 |  65.2 |
| checkpoint_100  |         1220000 | 100   |

## 3. Layers Available
First, Middle, Last for all checkpoints.

## 4. Number of Experts
64 experts per layer.

## 5. Matrix Dimensions
64x64 distance matrices.

## 6. Definition
Empirical KL divergence (functional distance) measured on the calibration set (SHA256: c7b221ff...).

## 7. Comparability
Directly comparable. The identical calibration set was used across all checkpoints.
However, early checkpoints (10, 40, 70) only have 384 evaluated pairs out of 2016. Checkpoint 100 has 2016 pairs. 

## 8. Alignment
Identities are strictly aligned because the model architecture and expert indexing in OLMoE are fixed during training.

## 9. Routing/Specialization Stats
MISSING - NOT FOUND in Exp3C data.

## 10. Missing Checkpoints/Data
No intermediate checkpoints other than 10%, 40%, 70%, 100%. Routing stats missing. Early checkpoints are sparse (19% coverage).

## 11. Leakage
Checkpoints were evaluated strictly sequentially in training order.

## 12. Input Hashes
See checkpoint_dataset.csv for exact artifact hashes.
