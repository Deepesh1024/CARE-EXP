import torch
x = torch.randn(3)
w = torch.randn(3, 3)
y = torch.nn.functional.linear(x, w)
print("y is leaf?", y.is_leaf)
print("y requires grad?", y.requires_grad)

y.requires_grad_(True)
print("y requires grad after?", y.requires_grad)

z = y * 2
loss = z.sum()
loss.backward()

print("y.grad:", y.grad)
