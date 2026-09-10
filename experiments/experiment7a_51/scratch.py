import torch

# Simulate hidden_states with no grad
hidden_states = torch.zeros(5, 4)

# Create final_hidden_states with no grad
final_hidden_states = torch.zeros_like(hidden_states)

# Simulate leaf_tensor with grad
leaf_tensor = torch.randn(2, 4).requires_grad_(True)

# Simulate out (requires grad because it depends on leaf_tensor)
out = leaf_tensor * 2.0

print(f"out requires grad: {out.requires_grad}")

# Index add
token_idx = torch.tensor([1, 3])
final_hidden_states = final_hidden_states.index_add(0, token_idx, out)

print(f"final_hidden_states requires grad: {final_hidden_states.requires_grad}")
