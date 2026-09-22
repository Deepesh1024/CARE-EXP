# Reproducing the CARE-COM External Benchmark

This document explains how to reproduce the external benchmark comparing CARE-COM against other state-of-the-art MoE expert compression methods on `allenai/OLMoE-1B-7B-0924`.

## 1. Environment Setup
Create a conda environment and install the required dependencies:

```bash
conda create -n care_benchmark python=3.10 -y
conda activate care_benchmark
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate pandas matplotlib seaborn pyyaml
```

## 2. Fetch External Repositories
We provide a bash script to clone all the required external repositories (REAP, REAM, PuzzleMoE):

```bash
chmod +x benchmark/setup_external.sh
./benchmark/setup_external.sh
```

## 3. Running the Benchmark

You can run all methods sequentially using:
```bash
python benchmark/run_benchmark.py --method all
```

Or you can run individual methods. This is highly recommended for parallelizing across multiple GPUs or tmux sessions:

**Internal Baselines:**
```bash
python benchmark/run_benchmark.py --method random --seed 42
python benchmark/run_benchmark.py --method care_static
python benchmark/run_benchmark.py --method care_adaptive
```

**External Methods:**
```bash
python benchmark/run_benchmark.py --method reap
python benchmark/run_benchmark.py --method hc_smoe
python benchmark/run_benchmark.py --method ream
python benchmark/run_benchmark.py --method puzzlemoe
```
*(Note: As documented in `METHOD_STATUS.md`, many external methods do not currently support OLMoE natively and will safely exit, logging their incompatibility).*

## 4. Aggregating Results & Plotting

Once the runs are complete, aggregate the raw `trajectory.json` files into clean CSV summaries:

```bash
python benchmark/aggregate_results.py
```

This will produce:
- `benchmark_results/summary/results.csv`
- `benchmark_results/summary/selection_analysis.csv`

Finally, generate all the figures required by the ICLR spec:
```bash
python benchmark/plot_results.py
```

The high-resolution plots will be saved to `benchmark_results/summary/plots/`.
