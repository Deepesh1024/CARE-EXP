# JetMoE-8B Inspection & Porting Report

## 1. Hardware & Environment
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)
- **PyTorch/CUDA:** Verified functional

## 2. Model Information
- **Model ID:** `jetmoe/jetmoe-8b`
- **Total Parameters:** ~8B
- **Active Parameters:** ~2.2B
- **Layers/Blocks:** 24
- **Experts per MoE:** 8 (2 active per token)

## 3. JetMoE Architecture Breakdown
- **MLP MoE Location:** `model.layers.X.mlp`
- **MoA Location:** `model.layers.X.self_attention.experts`

## 4. Expert Tensor Structure (MLP Only)
- **Expert Tensor Shapes:**
  - `input_linear.weight`: `[8, 11264, 2048]` (Batched SwiGLU: W1 and W3 concatenated)
  - `output_linear.weight`: `[8, 2048, 5632]` (Batched W2)
- **Router Tensor Shapes:** `router.layer.weight`: `[8, 2048]`
- **Storage Strategy:** JetMoE stores experts as **batched tensors** (using `JetMoeParallelExperts`) rather than as an `nn.ModuleList` of independent linear layers.

## 5. Routing Statistics
- **Token counts per expert:** Captured successfully via our forward hooks on `torch.nn.Linear` layers named `router`.

## 6. Baseline Performance
- **Baseline PPL:** 7.5085
- **Peak VRAM:** 16.26 GB (Fits perfectly on the 24GB RTX 4090)
- **Tokens/sec:** 11,221.49

## 7. Sanity Checks (Ablation & Merge)
- **Base PPL (10k tokens):** 8.5236
- **Ablated Expert 0 PPL:** 8.6880 (+0.1644 degradation)
- **Merged Expert 0 & 1 PPL:** 8.6443 (+0.1207 degradation)
- *Conclusion:* Physical merging works and is mathematically superior to ablation (less degradation).

## 8. Architectural Modifications Required for CARE
Based on the inspection, determine:
- [x] **Does merging require router modification?** Not strictly required for the physical weights, but CARE needs to reassign routing distributions.
- [x] **Does top-k indexing need modification?** No, but probabilities must be aggregated.
- [x] **Are tensor shapes compatible with a direct $W_{merged} = (W_i + W_j)/2$?** Yes, but because they are **batched**, we must slice dimension `0`. For example, `W_merged = (input_linear.weight[i] + input_linear.weight[j]) / 2`. To permanently compress the model, we will need to create a new tensor of shape `[7, 11264, 2048]` and concatenate the unmerged experts with the merged expert.

## 9. Next Recommended Implementation Step
Now that we have fully verified JetMoE's batched tensor layout, memory footprint, and merge behavior, the next step is to create a JetMoE-specific subclass or wrapper in our `benchmark/OLMoE-1B-7B/methods/care_com.py` that overrides the weight retrieval and replacement logic to use batched tensor slicing (`torch.cat()`) instead of `.pop()` on a `ModuleList`.
