# Merge Operator Specification

Based on the expert architecture of OLMoE, the experts are batched into a single `OlmoeExperts` module containing stacked tensors for all 64 experts. The merge operator must average the slices corresponding to the two experts in the first dimension.

## Tensors to Merge
1. `gate_up_proj` (shape `[num_experts, 2048, 2048]`)
2. `down_proj` (shape `[num_experts, 2048, 1024]`)

## Pseudocode
```python
def merge_experts(expert_module, expert_i_idx, expert_j_idx):
    '''
    expert_module is an instance of OlmoeExperts.
    '''
    with torch.no_grad():
        for param_name in ["gate_up_proj", "down_proj"]:
            w = getattr(expert_module, param_name) # w has shape [N, out_dim, in_dim]
            
            # Extract the weights for the two experts
            w_i = w[expert_i_idx]
            w_j = w[expert_j_idx]
            
            # Average the weights
            w_new = (w_i + w_j) / 2.0
            
            # Assign the merged weights back to the surviving expert (e.g., i)
            w[expert_i_idx].copy_(w_new)
            
            # The tensor for expert_j_idx becomes redundant and should be removed 
            # to actually reduce the parameter count. This requires creating a new 
            # Parameter with shape [num_experts - 1, out_dim, in_dim] and copying
            # all rows except expert_j_idx.
```

There are no expert-specific biases or normalization parameters in this architecture.
