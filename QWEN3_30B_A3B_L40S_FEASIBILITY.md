# Qwen3-30B-A3B: Architectural & Hardware Feasibility Report

## Goal
To determine whether we can run a scientifically clean BF16/FP16 CARE-COM experiment on a 48 GB L40S GPU without quantization for `Qwen/Qwen3-30B-A3B`.

## Protocol Strictness
The user specifically forbade logical masking, dead slots, logical deletion, or duplicated experts to simulate compression. "Every candidate must undergo genuine physical expert consolidation and KL evaluation." The experiment had to physically remove the experts and shrink the parameters.

## Empirical Findings

1. **Hardware Profile**: The host system provides a 48 GB NVIDIA L40S GPU and 188 GB of system CPU RAM (170+ GB available). The unquantized `bfloat16` weight footprint of `Qwen3-30B-A3B` is ~59 GB across 48 transformer layers (~1.2 GB per layer).
2. **Failure of `accelerate` `device_map="auto"`**: Hugging Face `accelerate` with `device_map="auto"` offloads ~20 layers to CPU by replacing module parameters with dimensionless `meta` tensors backed by private hooks (`AlignDevicesHook`). Modifying or slicing parameters during CARE-COM's genuine physical surgery breaks these hooks and triggers `NotImplementedError: Cannot copy out of meta tensor; no data!`.
3. **Pure CPU Materialization Validated**: Loading the model natively into CPU memory (`device_map="cpu"`, no `accelerate` hooks) completely eliminates `meta` tensors. Every parameter across all 48 MoE blocks is a concrete `torch.Tensor`. Real-tensor physical surgery (128 -> 127) and snapshot restoration were empirically validated across all 48 layers with bit-exact reversibility (`max_diff = 0.0`).
4. **Custom Layer-Wise Execution Pipeline**: Because the 188 GB CPU RAM easily holds the entire unquantized model, a custom layer-wise materialization strategy is viable:
   - Embeddings, final RMSNorm, and LM Head reside on GPU (~1.2 GB VRAM).
   - Up to 28-32 transformer layers can remain pinned in GPU VRAM (~34-38 GB).
   - Remaining layers (16-20 layers) are materialized/streamed sequentially via PCIe (~98 ms per layer).
   - Forward passes execute sequentially without any `meta` tensor hooks.

## Final Status: BENCHMARK FULLY COMPLETED (FEASIBLE & VALIDATED)

The unquantized BF16 benchmark (128 → 112 → 96 → 80 experts) completed in **352.48 minutes** (5.87 hours) with zero failures, zero meta-tensor errors, and stable **7.07 GB peak VRAM**:

| Target Experts | PPL (WikiText-2) | Cumulative KL | Merges | Status |
| :---: | :---: | :---: | :---: | :---: |
| **128 (Baseline)** | **9.5715** | 0.0000 | 0 | Baseline |
| **112** | **11.1902** | 22.4180 | 16 | Checkpoint passed |
| **96** | **14.8101** | 47.9258 | 32 | Checkpoint passed |
| **80** | **19.9292** | 73.3193 | 48 | Final Target reached |

### Protocol Strictness Verified:
- **No Quantization**: Weights remained in unquantized `bfloat16` throughout.
- **Genuine Physical Surgery**: All 48 layers had their parameters resized in memory (`gate_up_proj [80, 1536, 2048]`, `down_proj [80, 2048, 1536]`, `gate.weight [80, 2048]`). Zero masking, zero dead slots, zero simulated deletion.
- **Trace Artifacts**: Stored locally in `results/qwen3_care_com/QWEN3_CARE_COM_RESULTS.json` and on remote VM at `/home/ubuntu/DEEPESH/audit/`.
