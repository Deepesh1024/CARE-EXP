# OLMoE Architecture Inspection

## Model Configuration
- **Model Class:** OlmoeForCausalLM
- **MoE Block Class:** OlmoeSparseMoeBlock
- **Expert Module Class:** OlmoeExperts (Batched expert representations)
- **Router Class:** OlmoeTopKRouter
- **Expert Count:** 64
- **Hidden Size:** 2048
- **Intermediate Size:** 1024
- **Routing:** num_experts_per_tok = 8

## Expert Tensors
The `OlmoeExperts` class holds all 64 experts in single batched tensors. The first dimension corresponds to the expert index.

| Component | Tensor | Shape | Expert-specific? | Mergeable? | Notes |
|---|---|---|---|---|---|
| Expert (Batched) | `gate_up_proj` | `[64, 2048, 2048]` | Yes | Yes | Dimension 0 is the expert index. 2048 = 2 * 1024 (intermediate_size) |
| Expert (Batched) | `down_proj` | `[64, 2048, 1024]` | Yes | Yes | Dimension 0 is the expert index. |

No biases or layer norms were found inside the expert module tensors.

## Router Tensors
| Component | Tensor | Shape | Notes |
|---|---|---|---|
| Router | `weight` | `[64, 2048]` | Shape is `[num_experts, hidden_size]`. Determines routing probabilities. |
