# Router Merge Specification

The `OlmoeTopKRouter` contains a single linear weight matrix mapping from the hidden dimension (2048) to the number of experts (64).

## Tensors to Merge
- `weight` (shape `[num_experts, hidden_size]`)

## Pseudocode
```python
def update_router(router, expert_i_idx, expert_j_idx):
    '''
    router is an instance of OlmoeTopKRouter.
    '''
    with torch.no_grad():
        w = router.weight # shape [N, 2048]
        
        # Extract router weights for the two experts
        w_i = w[expert_i_idx]
        w_j = w[expert_j_idx]
        
        # Average the weights
        w_new = (w_i + w_j) / 2.0
        
        # Assign back to surviving index
        w[expert_i_idx].copy_(w_new)
        
        # Remove the row for expert_j_idx from the weight matrix to match
        # the new number of experts. This requires creating a new Parameter.
        
        # Additionally, any num_experts properties in the router or config
        # must be decremented by 1.
```

The router calculates logits via a linear projection `F.linear(hidden_states, self.weight)`. Averaging the weights for $i$ and $j$ is equivalent to averaging their routing logits. Since the experts are being averaged, averaging their router weights gives the merged expert a routing propensity that is the mean of the original two experts' propensities.
