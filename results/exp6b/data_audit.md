# Experiment 6B  Data Audit
**Generated:** 2026-08-16T15:26:42.458694
**Model:** allenai/OLMoE-1B-7B-0924
**Device:** cuda:0

## 1. Exp3C Checkpoint Oracle Distance Matrices

| Checkpoint | Layer | Pairs | CSV Hash (first 12) |
|---|---|---|---|
| checkpoint_10 | first | 384 | 23ad593c2950 |
| checkpoint_10 | middle | 384 | fd058a8c2fcb |
| checkpoint_10 | last | 384 | 0389308607ce |
| checkpoint_40 | first | 384 | 7ad733300f86 |
| checkpoint_40 | middle | 384 | 58a08a613e9e |
| checkpoint_40 | last | 384 | 00e122e0b465 |
| checkpoint_70 | first | 384 | ccf79630e0e0 |
| checkpoint_70 | middle | 384 | 48cd122fde32 |
| checkpoint_70 | last | 384 | 173c64b25c9e |
| checkpoint_100 | first | 2016 | 4f3bd7ec8949 |
| checkpoint_100 | middle | 2016 | 8a1d41593f26 |
| checkpoint_100 | last | 2016 | d99521a015c3 |

## 2. Exp3B q-Value Ranking

- Primary q: **4** (selected in Exp3B, fixed in Exp4)
- Secondary q: **6** (second-best across middle+last layers)
- Tertiary q: **3** (second-best for first layer)

## 3. Router Telemetry Availability

- Historical telemetry exists: **False**
- Can reconstruct from checkpoint + calibration: **True**
- Method: Forward pass of calibration dataset through each checkpoint revision. Router hooks capture logits, Top-k indices, probabilities, and input hidden states. This is deterministic and exactly reproducible because the calibration dataset is frozen (SHA256 verified) and inference is in eval mode with no dropout.

## 4. Calibration Dataset

- Path: `/home/sandlogic/LINGO/PROJECTS/Experiments-V3/experiments/experiment3c/data/calibration/calibration_3c_wikitext.pt`
- Exists: True
- SHA256 match: **True**

## 5. Upstream Experiments

- Exp4 final_report.json: L
- Exp6A prediction_results.csv: L

## 6. System Resources

- CUDA: True
- GPU: NVIDIA GeForce RTX 4090
- GPU Memory: 25.3 GB
