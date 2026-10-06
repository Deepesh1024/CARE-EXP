#!/bin/bash
set -e

echo "============================================================"
echo "Experiment 12: Phi-3.5-MoE Compression Benchmark"
echo "============================================================"
echo ""
echo "STEP 1: Run architecture audit (REQUIRED before benchmark)"
echo "============================================================"

python experiments/experiment12/phi_audit.py

# Check if audit passed
AUDIT_RESULT=$(python3 -c "
import json
with open('experiments/experiment12/results/audit_report.json') as f:
    r = json.load(f)
print('PASS' if r.get('passed') else 'FAIL')
")

if [ "$AUDIT_RESULT" != "PASS" ]; then
    echo ""
    echo "❌ AUDIT FAILED. Do not proceed to benchmark."
    echo "   Fix the stop conditions in audit_report.json first."
    exit 1
fi

echo ""
echo "✅ Audit passed."
echo ""
echo "STEP 2: Run full compression benchmark"
echo "============================================================"
python experiments/experiment12/run_benchmark.py

echo ""
echo "============================================================"
echo "Benchmark complete. Results in experiments/experiment12/results/"
echo "============================================================"
