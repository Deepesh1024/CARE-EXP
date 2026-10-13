#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$EXP_ROOT"

echo "=========================================================================="
echo "CARE-COM Exp 16 — Cross-Architecture Reasoning Evaluation on RTX 4090"
echo "=========================================================================="
echo "Working directory: $(pwd)"
echo "CUDA Devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'N/A')"
echo ""

# Phase 2: Smoke tests
echo "--- [PHASE 2] Running Smoke Tests ---"
python3 exp16_cross_arch_reasoning/run_smoke_tests.py

# Phase 3: JetMoE-8B Pilot
echo ""
echo "--- [PHASE 3] Running JetMoE-8B Reasoning Pilot (8, 7, 6, 4 experts) ---"
python3 exp16_cross_arch_reasoning/run_jetmoe_reasoning.py

# Phase 4: Phi-3.5-MoE Pilot
echo ""
echo "--- [PHASE 4] Running Phi-3.5-MoE Reasoning Pilot (16, 14, 12, 10, 8 experts) ---"
python3 exp16_cross_arch_reasoning/run_phi_reasoning.py

# Phase 5 & 6: Aggregate & Report
echo ""
echo "--- [PHASE 5 & 6] Aggregating Results & Generating Scientific Report ---"
python3 exp16_cross_arch_reasoning/aggregate_and_report.py

echo ""
echo "=========================================================================="
echo "EXP 16 FULL EVALUATION WORKFLOW COMPLETED SUCCESSFULLY!"
echo "=========================================================================="
