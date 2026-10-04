#!/bin/bash

# Experiment 9 Execution Pipeline
# -------------------------------
# This script runs the entire Experiment 9 pipeline sequentially.

set -e # Exit immediately if a command exits with a non-zero status

# Set working directory to the project root
cd "$(dirname "$0")/../.."

echo "=========================================================="
echo "Starting Experiment 9 Pipeline..."
echo "=========================================================="

echo -e "\n[1/3] Extracting Controls (Usage & RW-L2) on Split A..."
python experiments/experiment9/extract_controls.py

echo -e "\n[2/3] Training Models and Evaluating on Split B..."
python experiments/experiment9/run_exp9.py

echo -e "\n[3/3] Computing Statistics and Plotting..."
python experiments/experiment9/plot_exp9.py

echo -e "\n=========================================================="
echo "Pipeline Complete! Results saved to:"
echo "  - experiments/experiment9/results/table.md"
echo "  - experiments/experiment9/results/model_comparison.png"
echo "=========================================================="
