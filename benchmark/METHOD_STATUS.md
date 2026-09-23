# Method Status for OLMoE

During the construction of the CARE-COM External Benchmark, we analyzed the compatibility of several external MoE compression techniques with the `allenai/OLMoE-1B-7B-0924` model. As per the benchmark protocol, any external method that could not naturally ingest the OLMoE architecture without fabricating adaptations was documented as incompatible.

| Method | Status | Notes |
|:---|:---|:---|
| **Random (Internal Baseline)** | ✅ Fully Supported | Wrapped successfully. |
| **Static CARE-COM** | ✅ Fully Supported | Wrapped successfully. |
| **Adaptive CARE-COM** | ✅ Fully Supported | Wrapped successfully. |
| **REAP** | ✅ COMPATIBLE_AND_VALIDATED | Validated OLMoE compatibility adapters. |
| **HC-SMoE** | ✅ COMPATIBLE_AND_VALIDATED | Validated OLMoE compatibility adapters. |
| **M-SMoE** | ✅ COMPATIBLE_AND_VALIDATED | Validated OLMoE compatibility adapters. |
| **Sub-MoE** | ✅ COMPATIBLE_AND_VALIDATED | Validated OLMoE compatibility adapters. |
| **REAM** | ❌ Not Reproduced on OLMoE | Relies on hardcoded Mistral/Mixtral block configurations. |
| **PuzzleMoE** | ❌ Not Reproduced on OLMoE | Requires explicit modeling modifications specific to standard LLMs. |

*Note: For the external methods marked ❌, their execution wrappers gracefully catch the `ImportError` or `KeyError` and log the incompatibility to `benchmark_results/<method>/error.log`.*
