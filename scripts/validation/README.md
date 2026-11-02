# Real-Model Validation Harness for CARE-MoE

This directory contains a standalone, self-contained validation harness that
loads **OLMoE-1B-7B-0924** on a real GPU and tests all six compression methods:
CARE-Static, CARE-Adaptive, Sub-MoE, REAP-Prune, HC-SMoE, and MC-SMoE.

> [!IMPORTANT]
> **This harness is designed for an NVIDIA RTX 4090 VM** (or any CUDA machine
> with ≥ 16 GB VRAM and ≥ 32 GB CPU RAM). It will not run on the authoring
> Mac without significant changes.

---

## Files

| File | Purpose |
|:---|:---|
| `validate_real_olmoe.py` | Main Python validation script |
| `run_validation.sh` | Shell wrapper: arg parsing, preflight, logging |
| `README.md` | This file |

---

## Quickstart

### 1. Transfer the project to the VM

```bash
# From your Mac:
rsync -avz --exclude '.git' --exclude '__pycache__' \
    /path/to/Experiments-V3/ user@vm-host:/workspace/Experiments-V3/
```

Or use the pre-built archive from the previous session:

```bash
scp sync_to_remote.tar.gz user@vm-host:/workspace/
ssh user@vm-host "cd /workspace && tar xzf sync_to_remote.tar.gz"
```

### 2. Install dependencies on the VM

```bash
# PyTorch (adjust CUDA version as needed)
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Core packages
pip install transformers datasets scipy scikit-learn numpy pandas tqdm psutil
```

### 3. Run preflight only (sanity check before downloading the model)

```bash
cd /workspace/Experiments-V3
bash scripts/validation/run_validation.sh --preflight-only
```

### 4. Run a single-method smoke test

```bash
bash scripts/validation/run_validation.sh --methods care-adaptive
```

### 5. Run all six methods

```bash
bash scripts/validation/run_validation.sh --methods all
```

---

## CLI Reference

### Shell wrapper (`run_validation.sh`)

| Flag | Default | Description |
|:---|:---|:---|
| `--methods METHOD...` | `all` | Space-separated list of methods to test |
| `--dtype DTYPE` | `bfloat16` | Model dtype: `float32`, `float16`, `bfloat16` |
| `--device DEVICE` | `cuda:0` | PyTorch device string |
| `--max-cal N` | `8` | Number of calibration batches |
| `--seq-len N` | `256` | Sequence length per calibration batch |
| `--model-id ID` | `allenai/OLMoE-1B-7B-0924` | HuggingFace model ID |
| `--output-dir DIR` | `reports/REAL_MODEL_VALIDATION` | Output directory |
| `--keep-checkpoints` | off | Keep per-method compressed checkpoints |
| `--preflight-only` | off | Only run hardware/dependency checks |
| `--dry-run` | off | Print intended actions, no execution |
| `--help` | — | Show usage |

### Python script (`validate_real_olmoe.py`)

```text
python validate_real_olmoe.py [OPTIONS]

  --model-id ID                 HuggingFace model ID
  --output-dir DIR              Base output directory (timestamped run subdir created)
  --methods METHOD [METHOD ...] care-static care-adaptive sub-moe reap-prune hc-smoe mc-smoe
  --target-experts N            Target expert count (default: 63)
  --max-calibration-batches N   Number of calibration batches (default: 8)
  --max-sequence-length N       Sequence length (default: 256)
  --device DEVICE               cuda:0 or cpu
  --dtype {float32,float16,bfloat16}
  --seed N                      Random seed (default: 42)
  --keep-checkpoints            Retain per-method checkpoints
  --dry-run                     No model loading or compression
  --preflight-only              Stop after environment checks
```

---

## Output Structure

Each run creates a timestamped subdirectory:

```
reports/REAL_MODEL_VALIDATION/
└── run_20241022T120000Z/
    ├── environment.json         # Full hardware/software inventory
    ├── model_info.json          # Model introspection (shapes, param counts)
    ├── validation.log           # Verbose file log
    ├── results.json             # Machine-readable: all check records
    ├── metrics.json             # Runtime and memory summary
    ├── failure_details.json     # Errors only (empty dict on clean run)
    ├── summary.md               # Human-readable validation report
    ├── calibration/             # (Optional) saved calibration batches
    ├── care_static_results.json
    ├── care_adaptive_results.json
    ├── sub_moe_results.json
    ├── reap_prune_results.json
    ├── hc_smoe_results.json
    ├── mc_smoe_results.json
    └── checkpoints/             # Per-method compressed models (--keep-checkpoints only)
        ├── care_static/
        ├── care_adaptive/
        ├── sub_moe/
        ├── reap_prune/
        ├── hc_smoe/
        └── mc_smoe/
```

### `results.json` schema

Each element in the array is a single check record:

```json
{
  "check_id": "CARE-ADAPTIVE-005",
  "method": "care-adaptive",
  "check_name": "KL damage evaluation for all candidates",
  "status": "PASS",
  "start_time": "2024-10-22T12:01:05.123000Z",
  "end_time": "2024-10-22T12:01:12.456000Z",
  "duration_s": 7.333,
  "config": {},
  "expected": "...",
  "actual": "5 evaluations, all_finite=True",
  "error_msg": null,
  "traceback": null,
  "metrics": {"eval_count": 5, "all_finite": true},
  "artifact_paths": []
}
```

### Exit codes

| Code | Meaning |
|:---:|:---|
| `0` | All executed checks passed |
| `1` | One or more checks FAILED |
| `2` | Environment blocked validation (missing CUDA, failed model load, etc.) |
| `3` | Invalid arguments |

---

## Check Inventory

### Preflight (PREFLIGHT-001 to PREFLIGHT-012)

| ID | Check |
|:---|:---|
| PREFLIGHT-001 | OS and Python version |
| PREFLIGHT-002 | PyTorch version |
| PREFLIGHT-003 | Transformers version |
| PREFLIGHT-004 | CUDA availability |
| PREFLIGHT-005 | GPU hardware detection |
| PREFLIGHT-006 | nvidia-smi driver query |
| PREFLIGHT-007 | CPU RAM (via psutil) |
| PREFLIGHT-008 | Disk free space |
| PREFLIGHT-009 | Git commit and dirty-state |
| PREFLIGHT-010 | Key Python dependencies (scipy, sklearn, etc.) |
| PREFLIGHT-011 | Project import paths |
| PREFLIGHT-012 | HuggingFace model cache check |

### Model Loading (MODEL-001 to MODEL-003)

| ID | Check |
|:---|:---|
| MODEL-001 | Tokenizer loading |
| MODEL-002 | Model weight loading (dtype, device map, timing) |
| MODEL-003 | Baseline forward pass on fixed input |

### CARE-Static (CARE-STATIC-001 to CARE-STATIC-007)

| ID | Check |
|:---|:---|
| CARE-STATIC-001 | Model copy for isolation |
| CARE-STATIC-002 | Import capability.py, core.py |
| CARE-STATIC-003 | Capability extraction at t=0 (shape, finiteness) |
| CARE-STATIC-004 | Frozen ranking generation (all N*(N-1)/2 pairs sorted) |
| CARE-STATIC-005 | Physical merge: expert count drops by exactly 1 |
| CARE-STATIC-006 | Frozen ranking immutability after merge |
| CARE-STATIC-007 | Post-merge forward pass (finite logits) |
| CARE-STATIC-CKPT-* | Checkpoint save, reload, logit parity |

### CARE-Adaptive (CARE-ADAPTIVE-001 to CARE-ADAPTIVE-010)

| ID | Check |
|:---|:---|
| CARE-ADAPTIVE-001 | Model copy for isolation |
| CARE-ADAPTIVE-002 | Import CARE v21 modules |
| CARE-ADAPTIVE-003 | Capability extraction at t=0 |
| CARE-ADAPTIVE-004 | Candidate pool generation (K=5) |
| CARE-ADAPTIVE-005 | KL damage evaluated for every candidate |
| CARE-ADAPTIVE-006 | Argmin-damage candidate selected and committed |
| CARE-ADAPTIVE-007 | Capability state changes after merge (shape check) |
| CARE-ADAPTIVE-008 | Evidence of capability re-extraction (instrumentation note) |
| CARE-ADAPTIVE-009 | Post-merge forward pass |
| CARE-ADAPTIVE-010 | Multi-step check (NOT_RUN in smoke-test mode) |
| CARE-ADAPTIVE-CKPT-* | Checkpoint save, reload, logit parity |

### Sub-MoE (SUB-MOE-001 to SUB-MOE-005)

| ID | Check |
|:---|:---|
| SUB-MOE-001 | Import SubMoEEngine |
| SUB-MOE-002 | Model copy |
| SUB-MOE-003 | compress_block: Joint SVD + expert count reduction |
| SUB-MOE-004 | SVD execution fidelity note |
| SUB-MOE-005 | Post-compression forward pass |
| SUB-MOE-CKPT-* | Checkpoint save/reload |

### REAP-Prune (REAP-001 to REAP-006)

| ID | Check |
|:---|:---|
| REAP-001 | Import REAPSaliencyObserver / REAPPruneEngine |
| REAP-002 | Model copy |
| REAP-003 | Saliency observer accumulation (n_layers, n_tokens, finiteness) |
| REAP-004 | Physical pruning: expert count drops by 1 (no merging) |
| REAP-005 | Categorical distinction from merging methods (fidelity note) |
| REAP-006 | Post-pruning forward pass |
| REAP-CKPT-* | Checkpoint save/reload |

### HC-SMoE (HCSMOE-001 to HCSMOE-005)

| ID | Check |
|:---|:---|
| HCSMOE-001 | Import HCSMoEEngine |
| HCSMOE-002 | Model copy |
| HCSMOE-003 | Agglomerative compress_block: expert count check |
| HCSMOE-004 | Activation-vs-router similarity fidelity warning |
| HCSMOE-005 | Post-merge forward pass |
| HCSMOE-CKPT-* | Checkpoint save/reload |

### MC-SMoE (MCSMOE-001 to MCSMOE-005)

| ID | Check |
|:---|:---|
| MCSMOE-001 | Import MCSMoEEngine |
| MCSMOE-002 | Model copy |
| MCSMOE-003 | Hungarian permute + merge: expert count check |
| MCSMOE-004 | Stage 5 (low-rank) missing fidelity warning |
| MCSMOE-005 | Post-merge forward pass |
| MCSMOE-CKPT-* | Checkpoint save/reload |

---

## Scientific Caveats

> [!WARNING]
> A PASS status means **the code executed without error and the expert count
> decreased correctly**. It does **not** certify:
> - Algorithmic fidelity to the published paper
> - Correctness of the merged/pruned model's outputs
> - That the compression method outperforms or underperforms CARE-COM

> [!NOTE]
> Two methods carry explicit **partial-fidelity warnings** in their results:
>
> - **HC-SMoE**: Uses router weight cosine similarity as the pairwise distance
>   metric. The paper specifies activation-profile cosine distance on calibration
>   data. Label outputs as _HC-SMoE (router-proxy)_ in any publication.
>
> - **MC-SMoE**: Executes Stages 1–4 (grouping, dominant expert, Hungarian
>   matching, permuted merge). Stage 5 (low-rank SVD residual compression) is
>   **not implemented**. Label outputs as _MC-SMoE (Stages 1–4 only)_.

> [!CAUTION]
> Do not use the perplexity numbers from this smoke-test run (1 merge / 1 prune
> per method) as benchmark evidence. Full benchmark authorization and multi-seed
> evaluation are required before any comparative claim can be made.

---

## Memory Budget (RTX 4090, 24 GB VRAM)

| Stage | Estimated VRAM |
|:---|:---|
| OLMoE-1B-7B baseline (bfloat16) | ~4 GB |
| + deep copy for isolation | ~8 GB |
| + calibration + KL pass | ~10 GB |
| Peak (CARE-Adaptive, K=5 candidates) | ~12–14 GB |

If OOM errors occur, reduce `--max-cal` and `--seq-len`, or switch to `--dtype float16`.

---

## Troubleshooting

**`BLOCKED: No MoE blocks found`**
The model loader mapped the model across multiple devices and the named_modules()
traversal fails. Try `--device cpu` first to confirm the architecture is detected,
then re-run with `--device cuda:0`.

**`ImportError: cannot import name 'generate_candidate_pool'`**
Ensure `sys.path` includes the project root and the `experiments/` directory.
Run from the repository root, not from within `scripts/validation/`.

**`OutOfMemoryError during deep copy`**
Deep-copying a loaded bf16 model doubles peak VRAM. If this fails, reduce dtype
to fp16 or enable `low_cpu_mem_usage=True` (already set in model loading).

**`WikiText-2 load failed`**
The script falls back to four synthetic sentences. Calibration is still valid
for smoke-test purposes; saliency scores will differ from full-data calibration.

---

*This harness was generated by the Antigravity IDE as part of the CARE-MoE
research integrity task. All fidelity warnings should be preserved in any
downstream publication or supplementary material.*
