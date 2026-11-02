#!/usr/bin/env bash
# =============================================================================
# scripts/validation/run_validation.sh
# =============================================================================
# Preflight + validation runner for the CARE-MoE project on an NVIDIA RTX 4090
# (or any CUDA-capable VM). Runs validate_real_olmoe.py with sane defaults and
# captures both the terminal output and the structured JSON/Markdown reports.
#
# Usage:
#   bash run_validation.sh [OPTIONS]
#
# Options:
#   --methods METHOD [METHOD ...]   Methods to validate (default: all)
#   --dtype DTYPE                   Model dtype: float32 | float16 | bfloat16 (default: bfloat16)
#   --device DEVICE                 PyTorch device (default: cuda:0)
#   --max-cal N                     Number of calibration batches (default: 8)
#   --seq-len N                     Calibration sequence length (default: 256)
#   --model-id ID                   HuggingFace model ID (default: allenai/OLMoE-1B-7B-0924)
#   --output-dir DIR                Base output directory (default: repo/reports/REAL_MODEL_VALIDATION)
#   --keep-checkpoints              Retain compressed model checkpoints after validation
#   --preflight-only                Run only environment checks, skip model loading
#   --dry-run                       Print intended actions without executing
#   --help                          Show this message
# =============================================================================

set -euo pipefail

# ── Locate project root (this script lives in scripts/validation/) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
VALIDATE_SCRIPT="$SCRIPT_DIR/validate_real_olmoe.py"

# ── Defaults ──
METHODS="all"
DTYPE="bfloat16"
DEVICE="cuda:0"
MAX_CAL="8"
SEQ_LEN="256"
MODEL_ID="allenai/OLMoE-1B-7B-0924"
OUTPUT_DIR="$REPO_ROOT/reports/REAL_MODEL_VALIDATION"
EXTRA_FLAGS=""
PREFLIGHT_ONLY=0
DRY_RUN=0

# ── Colour helpers ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }
sep()     { echo "────────────────────────────────────────────────────────────"; }

# ── Argument parsing ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)
            shift; METHODS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                METHODS="$METHODS $1"; shift
            done
            METHODS="${METHODS# }"
            ;;
        --dtype)             DTYPE="$2";       shift 2 ;;
        --device)            DEVICE="$2";      shift 2 ;;
        --max-cal)           MAX_CAL="$2";     shift 2 ;;
        --seq-len)           SEQ_LEN="$2";     shift 2 ;;
        --model-id)          MODEL_ID="$2";    shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";  shift 2 ;;
        --keep-checkpoints)  EXTRA_FLAGS="$EXTRA_FLAGS --keep-checkpoints"; shift ;;
        --adaptive-steps)    EXTRA_FLAGS="$EXTRA_FLAGS --adaptive-steps $2"; shift 2 ;;
        --router-merge-mode) EXTRA_FLAGS="$EXTRA_FLAGS --router-merge-mode $2"; shift 2 ;;
        --preflight-only)    PREFLIGHT_ONLY=1; shift ;;
        --dry-run)           DRY_RUN=1;        shift ;;
        --help|-h)
            sed -E -n '/^# Usage:/,/^# ===*/{ s/^#[ ]?//; p; }' "$0"
            exit 0
            ;;
        *) fail "Unknown argument: $1"; exit 3 ;;
    esac
done

sep
echo ""
info "CARE-MoE Real-Model Validation Harness"
info "Repo root:  $REPO_ROOT"
info "Model:      $MODEL_ID"
info "Methods:    $METHODS"
info "Device:     $DEVICE"
info "dtype:      $DTYPE"
info "Cal batches: $MAX_CAL  |  Seq len: $SEQ_LEN"
echo ""
sep

# ── Step 1: Python reachability ──
info "Checking Python at: $($PYTHON --version 2>&1)"
if ! command -v "$PYTHON" &>/dev/null; then
    fail "Python interpreter not found: $PYTHON"
    fail "Set the PYTHON env variable, e.g.: PYTHON=python3.10 bash run_validation.sh"
    exit 2
fi

# ── Step 2: CUDA reachability ──
info "Checking CUDA..."
CUDA_STATUS=$($PYTHON -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "TORCH_MISSING")
if [ "$CUDA_STATUS" = "TORCH_MISSING" ]; then
    fail "PyTorch is not importable. Install it before running validation."
    exit 2
elif [ "$CUDA_STATUS" = "CPU" ]; then
    warn "CUDA is not available — model will run on CPU (very slow, expect OOM for large models)."
    DEVICE="cpu"
fi
info "Compute backend: $CUDA_STATUS"

# ── Step 3: Key dependencies ──
info "Checking key Python packages..."
MISSING=""
for pkg in transformers datasets scipy sklearn numpy pandas tqdm; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done
if [ -n "$MISSING" ]; then
    warn "Missing packages:$MISSING"
    warn "Run: pip install$MISSING"
fi

# ── Step 4: nvidia-smi ──
if command -v nvidia-smi &>/dev/null; then
    info "GPU(s) detected:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
        | awk '{print "    GPU " $0}'
else
    warn "nvidia-smi not found — skipping hardware inventory"
fi

sep

# ── Step 5: Build the Python command ──
FULL_CMD=(
    "$PYTHON" "$VALIDATE_SCRIPT"
    --model-id "$MODEL_ID"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --max-calibration-batches "$MAX_CAL"
    --max-sequence-length "$SEQ_LEN"
)

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    FULL_CMD+=(--preflight-only)
else
    if [ "$DRY_RUN" -eq 1 ]; then
        FULL_CMD+=(--dry-run)
    fi
    FULL_CMD+=(--methods)
    for m in $METHODS; do
        FULL_CMD+=("$m")
    done
fi

for flag in $EXTRA_FLAGS; do
    FULL_CMD+=("$flag")
done

info "Command: ${FULL_CMD[*]}"
sep

# ── Step 6: Create output directory and tee log ──
mkdir -p "$OUTPUT_DIR"
SHELL_LOG="$OUTPUT_DIR/run_validation_$(date -u +%Y%m%dT%H%M%SZ).shell.log"
info "Shell log: $SHELL_LOG"
echo ""

# ── Step 7: Execute ──
set +e
"${FULL_CMD[@]}" 2>&1 | tee "$SHELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

sep
if [ "$EXIT_CODE" -eq 0 ]; then
    success "Validation completed with no failures (exit 0)"
elif [ "$EXIT_CODE" -eq 1 ]; then
    fail "Validation completed but one or more checks FAILED (exit 1)"
elif [ "$EXIT_CODE" -eq 2 ]; then
    fail "Validation BLOCKED: environment or model loading failed (exit 2)"
elif [ "$EXIT_CODE" -eq 3 ]; then
    fail "Bad arguments (exit 3)"
else
    warn "Unexpected exit code: $EXIT_CODE"
fi

echo ""
info "Reports directory: $OUTPUT_DIR"
info "Shell log:         $SHELL_LOG"

# ── Step 8: Show the latest summary if available ──
LATEST_RUN=$(ls -dt "$OUTPUT_DIR"/run_* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/summary.md" ]; then
    echo ""
    sep
    info "Quick summary from $LATEST_RUN/summary.md:"
    sep
    head -80 "$LATEST_RUN/summary.md"
    echo ""
fi

exit "$EXIT_CODE"
