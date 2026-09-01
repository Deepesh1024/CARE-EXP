# TABLE 1: CAPABILITY INSTRUMENT / CARE-BENCH

**Source References:**
- `experiments/experiment6c/config.py`
- `results/exp6c/EXP6C_FINAL_AUDIT.md`
- `results/exp6d_rerun/exp6d/EXP6D_FINAL_REPORT.md`

| Parameter / Dimension | Definition / Description | Authoritative Source / Notes |
| :--- | :--- | :--- |
| **Model Architecture** | `OLMoE-1B-7B-0924` (16 layers, 64 experts/layer, 8 routed experts/token) | `config.py` |
| **Capability Space** | $C_i \in \mathbb{R}^{10}$ (10-dimensional empirical response space) | `EXP6D_FINAL_REPORT.md` |
| **Probe Dataset 1** | `cais/mmlu` | `config.py` |
| **Probe Dataset 2** | `ai2_arc` (ARC-Challenge) | `config.py` |
| **Calibration Budget** | Maximum 300 samples per semantic category | `config.py` |
| **Axis 1: Mathematics** | MMLU (College, High School, Elementary Math) | `config.py` |
| **Axis 2: Physics/Astro** | MMLU (College Physics, Astronomy) | `config.py` |
| **Axis 3: Bio/Medicine** | MMLU (Anatomy, College Med, Clinical Knowledge) | `config.py` |
| **Axis 4: CS/Engineering** | MMLU (College CS, Machine Learning) | `config.py` |
| **Axis 5: Law** | MMLU (Jurisprudence, Professional/International Law) | `config.py` |
| **Axis 6: History** | MMLU (High School European/World History) | `config.py` |
| **Axis 7: Philosophy/Logic** | MMLU (Philosophy, Formal Logic, Moral Disputes) | `config.py` |
| **Axis 8: Business/Econ** | MMLU (Econometrics, HS Macroeconomics, Business Ethics) | `config.py` |
| **Axis 9: Psychology/Soc** | MMLU (Sociology, Professional/HS Psychology) | `config.py` |
| **Axis 10: General Reasoning** | ARC-Challenge | `config.py` |
