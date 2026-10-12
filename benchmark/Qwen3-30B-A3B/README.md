# Qwen3-30B-A3B Benchmark (Experiment 13)

This directory contains the benchmark harnesses, validation tests, and documentation for **`Qwen/Qwen3-30B-A3B`** (30B parameters, 48 layers, 128 experts per layer).

## Files:
- **`Exp13_Section.md`**: Formal scientific report for Experiment 13, documenting the VRAM engineering breakthrough (sequential PCIe streaming), validation gates, and empirical findings.
- **`golden_test.py`**: Audited 128 $\rightarrow$ 127 single physical merge test with bit-level parameter restoration checks.
- **`run_benchmark.py`**: Sequential layer-streaming benchmark runner supporting CARE-Adaptive, REAP, Sub-MoE, and Random Merging.

## Related Results:
Full empirical outputs, checkpoints, logs, and comparative plots are located in:
`benchmark_results/Qwen3-30B-A3B/`
