#!/bin/bash
set -e

echo "Setting up external benchmark repositories..."

mkdir -p benchmark/external
cd benchmark/external

# REAP / HC-SMoE / M-SMoE / Sub-MoE
if [ ! -d "reap" ]; then
    echo "Cloning REAP..."
    git clone https://github.com/CerebrasResearch/reap.git
else
    echo "REAP already cloned."
fi

# REAM
if [ ! -d "moe-expert-compress" ]; then
    echo "Cloning REAM..."
    git clone https://github.com/puwaer/moe-expert-compress.git
else
    echo "REAM already cloned."
fi

# PuzzleMoE
if [ ! -d "PuzzleMoE" ]; then
    echo "Cloning PuzzleMoE..."
    git clone https://github.com/Supercomputing-System-AI-Lab/PuzzleMoE.git
else
    echo "PuzzleMoE already cloned."
fi

echo "Setup complete!"
