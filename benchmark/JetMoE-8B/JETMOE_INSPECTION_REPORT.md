# JetMoE-8B Inspection & Porting Report

## 1. Hardware & Environment
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)
- **PyTorch:** (Fill from phase1 output)
- **CUDA:** (Fill from phase1 output)

## 2. Model Information
- **Model ID:** `jetmoe/jetmoe-8b`
- **Total Parameters:** ~8B
- **Active Parameters:** ~2.2B
- **Layers/Blocks:** 24
- **Experts per MoE:** 8 (2 active per token)

## 3. JetMoE Architecture Breakdown
*(Fill from `jetmoe_architecture.json`)*
- **MLP MoE Location:** e.g., `model.layers.X.mlp`
- **MoA Location:** e.g., `model.layers.X.self_attn`

## 4. Expert Tensor Structure (MLP Only)
*(Fill from `jetmoe_tensors_layer0.json`)*
- **Expert Tensor Shapes:**
- **Router Tensor Shapes:**
- **Are experts stored as batched tensors or nn.ModuleList?** (Document here)

## 5. Routing Statistics
*(Fill from `jetmoe_routing_stats.json`)*
- **Token counts per expert:** 
- **Load Imbalance observations:**

## 6. Baseline Performance
*(Fill from `baseline_metrics.json`)*
- **Baseline PPL:**
- **Peak VRAM:**
- **Tokens/sec:**

## 7. Sanity Checks (Ablation & Merge)
*(Fill from `expert_ablation_sanity.json`)*
- **Base PPL (10k tokens):**
- **Ablated Expert 0 PPL:** (Expected to degrade)
- **Merged Expert 0 & 1 PPL:** (Expected to degrade less than ablation)

## 8. Architectural Modifications Required for CARE
Based on the inspection, determine:
- [ ] Does merging require router modification?
- [ ] Does top-k indexing need modification?
- [ ] Are tensor shapes compatible with a direct $W_{merged} = (W_i + W_j)/2$ in a smaller $N-1$ tensor? (If batched, we must slice and concat. If ModuleList, we can replace or remove elements).

## 9. Next Recommended Implementation Step
*(To be determined after reviewing this report on the VM)*
