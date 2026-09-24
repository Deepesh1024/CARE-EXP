# Reproducibility Statement

The CARE-COM codebase and benchmark harness will be provided as anonymous supplementary material. 

**Models:** 
All experiments utilize the open-source `allenai/OLMoE-1B-7B-0924` model. 

**Data:**
Calibration and evaluation are performed on deterministic sequential chunks of `WikiText-2`.

**Scripts:**
- Experiment 4: `reconstruct_exp4.py` and `audit_analysis.py` (available in `audit/`)
- Benchmark: `python benchmark/run_benchmark.py --method all`
- Plotting: `python benchmark/plot_results.py`
