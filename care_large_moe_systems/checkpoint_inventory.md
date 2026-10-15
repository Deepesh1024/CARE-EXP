# Systems Benchmark: MoE Checkpoint & Trajectory Inventory

**Document Type:** Formal Artifact & Checkpoint Audit  
**Target Architectures:** `allenai/OLMoE-1B-7B-0924` (64 experts), `Qwen/Qwen3-30B-A3B` (128 experts), `jetmoe/jetmoe-8b` (8 experts)  
**Location:** [`care_large_moe_systems/checkpoint_inventory.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/checkpoint_inventory.md)

---

## 1. Checkpoint Verification Policy

Sparse MoE models possess enormous parameter footprints (14 GB for OLMoE-1B-7B in BF16, 59 GB for Qwen3-30B-A3B in BF16). Storing monolithic static `.pt` or `.safetensors` files for every intermediate compression step ($E \in \{64, 60, 56, 48, 40, 32\}$ and $E \in \{128, 112, 96, 90, 80, 64\}$) would consume $> 500\text{ GB}$ of storage, which exceeds standard disk allocations.

Consequently, CARE-COM implements a **deterministic trajectory replay protocol**:
- The pristine base model weights are cached once from HuggingFace.
- Every physical merge operation is recorded as an exact, deterministic sequence of merged index pairs $(i, j)$ in structured JSON trajectory files.
- Applying the recorded merge sequence to the base model reproduces the exact bit-level compressed checkpoint in seconds without running expensive search.

Per user instructions:
> *"First inspect the repository to identify which checkpoints actually exist. Do not regenerate missing checkpoints without authorization."*

---

## 2. Primary Model: OLMoE-1B-7B (64 Experts Base)

- **HuggingFace Repository:** `allenai/OLMoE-1B-7B-0924`
- **Architecture:** 16 decoder layers, hidden dimension $H = 2048$, intermediate dimension $I = 1024$ (fused gate+up: 2048), 64 experts per layer, top-$k = 8$.
- **Precision:** `bfloat16` (native, unquantized).
- **Base Footprint:** ~6.92B total parameters (~13.84 GB VRAM in BF16). Active parameters per token: ~1.28B (~2.56 GB equivalent FLOPs).

### Checkpoint & Trajectory Availability Matrix

| Target Expert Count ($E$) | Active Compression Ratio | Trajectory Artifact Location | Checkpoint Status | Baselines Available |
|:---:|:---:|---|:---:|---|
| **64** | 0.0% (Base) | Native HF `allenai/OLMoE-1B-7B-0924` | **AVAILABLE** | Baseline |
| **60** | -6.25% (4 merges) | [`benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json#L50-L100) | **REPRODUCIBLE** | CARE-Static, Sub-MoE, MC-SMoE, HC-SMoE, Random (seeds 42, 43, 44) |
| **56** | -12.5% (8 merges) | [`benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json#L101-L200) | **REPRODUCIBLE** | CARE-Static, Sub-MoE, MC-SMoE, HC-SMoE, Random (seeds 42, 43, 44) |
| **48** | -25.0% (16 merges) | [`benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/OLMoE-1B-7B/care_adaptive/trajectory.json#L201-L676) | **REPRODUCIBLE** | CARE-Static, Sub-MoE, MC-SMoE, HC-SMoE, Random (seeds 42, 43, 44) |
| **40** | -37.5% (24 merges) | None in `benchmark_results/` | **MISSING** | Requires user authorization to generate |
| **32** | -50.0% (32 merges) | None in `benchmark_results/` | **MISSING** | Requires user authorization to generate |

---

## 3. Secondary Model: Qwen3-30B-A3B (128 Experts Base)

- **HuggingFace Repository:** `Qwen/Qwen3-30B-A3B`
- **Architecture:** 48 decoder layers, hidden dimension $H = 2048$, intermediate dimension $I = 768$ (fused gate+up: 1536), 128 experts per layer, top-$k = 8$.
- **Precision:** `bfloat16` (native, unquantized).
- **Base Footprint:** ~30.7B total parameters (~61.4 GB VRAM in BF16). Active parameters per token: ~3.3B (~6.6 GB equivalent FLOPs).
- **Physical Cache Status:** Fully cached on remote L40S VM at `~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B`.

### Checkpoint & Trajectory Availability Matrix

| Target Expert Count ($E$) | Active Compression Ratio | Trajectory Artifact Location | Checkpoint Status | Feasibility Regime on L40S (48 GB) |
|:---:|:---:|---|:---:|---|
| **128** | 0.0% (Base) | Cached HF `Qwen/Qwen3-30B-A3B` | **AVAILABLE** | **CPU Offload / Layer Stream** (61.4 GB > 48 GB VRAM) |
| **112** | -12.5% (16 merges) | [`audit/benchmark_results_112.json`](file:///home/ubuntu/DEEPESH/audit/benchmark_results_112.json) | **REPRODUCIBLE** | **CPU Offload / Layer Stream** (54.1 GB > 48 GB VRAM) |
| **96** | -25.0% (32 merges) | [`audit/benchmark_results_96.json`](file:///home/ubuntu/DEEPESH/audit/benchmark_results_96.json) | **REPRODUCIBLE** | **Borderline / Offload** (46.8 GB model + KV cache) |
| **90** | -29.7% (38 merges) | [`exp14_reasoning/pilot_reasoning_90exp_report.json`](file:///home/ubuntu/DEEPESH/exp14_reasoning/pilot_reasoning_90exp_report.json) | **REPRODUCIBLE** | **GPU Resident (Careful)** (~44.1 GB model) |
| **80** | -37.5% (48 merges) | [`audit/benchmark_results_80.json`](file:///home/ubuntu/DEEPESH/audit/benchmark_results_80.json) | **REPRODUCIBLE** | **PURE GPU RESIDENT** (~39.3 GB model fits in 48 GB) |
| **64** | -50.0% (64 merges) | `/dev/shm/exp14_reasoning/care_results_64.json` | **REPRODUCIBLE** | **PURE GPU RESIDENT** (~32.1 GB model fits with 16 GB free) |

---

## 4. Optional Validation Model: JetMoE-8B (8 Experts Base)

- **HuggingFace Repository:** `jetmoe/jetmoe-8b`
- **Architecture:** 24 decoder layers, 8 experts per layer (mixture of attention + mixture of MLP), top-$k = 2$.
- **Precision:** `bfloat16`.
- **Base Footprint:** ~8.4B total parameters (~16.8 GB in BF16).
- **Existing Results:** Available in [`experiments/exp15_merging_benchmarks/results/jetmoe_results.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp15_merging_benchmarks/results/jetmoe_results.csv) at $E \in \{8, 7, 6, 4\}$.
