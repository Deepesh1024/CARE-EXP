#!/bin/bash
set -e

echo "============================================================"
echo "Starting Experiment 11: Capability Manifold Validation"
echo "============================================================"

# Ensure PYTHONPATH is set to the root directory
export PYTHONPATH="$(dirname "$(dirname "$(dirname "$0")")"):$PYTHONPATH"

# Run the validation script
echo "Running validate_manifold.py..."
python experiments/experiment11/validate_manifold.py

echo "Done!"
