#!/bin/bash
set -e

echo "============================================================"
export PYTHONPATH=.:$PYTHONPATH
echo "Starting Experiment 10: Functional Compressibility"
echo "============================================================"

echo ""
echo "[1/2] Extracting raw outputs, computing SVDs, and calculating FRC..."
python experiments/experiment10/extract_compressibility.py

echo ""
echo "[2/2] Generating final report and plots..."
python experiments/experiment10/plot_exp10.py

echo ""
echo "Done! Check experiments/experiment10/results/report.md"
