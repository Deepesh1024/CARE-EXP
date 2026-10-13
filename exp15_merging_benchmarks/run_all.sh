#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================="
echo "EXPERIMENT 15: HC-SMoE & MC-SMoE BENCHMARK SUITE (RTX 4090)"
echo "================================================================="
echo "[1/2] Launching Phase 2: JetMoE-8B Benchmark..."
python run_jetmoe_benchmarks.py

echo ""
echo "[2/2] Launching Phase 3: Phi-3.5-MoE Benchmark..."
python run_phi_benchmarks.py

echo ""
echo "================================================================="
echo "ALL EXP 15 BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Results available in: $DIR/results/"
echo "================================================================="
