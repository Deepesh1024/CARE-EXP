import torch
x = torch.randn(3, requires_grad=True)
y = torch.zeros_like(x)
print("y requires_grad:", y.requires_grad)

try:
    y.index_add_(0, torch.tensor([0]), x[:1])
    print("Success! y requires_grad after:", y.requires_grad)
except Exception as e:
    print("Error:", e)
