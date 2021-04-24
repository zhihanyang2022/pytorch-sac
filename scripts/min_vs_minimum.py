import torch

a = torch.tensor([[1], [-1], [1]], dtype=torch.float, requires_grad=True)
b = torch.tensor([[-1], [1], [-1]], dtype=torch.float, requires_grad=True)

c = torch.sum(torch.min(a, b))

c.backward()

print(a.grad)
print(b.grad)

a = torch.tensor([[1], [-1], [1]], dtype=torch.float, requires_grad=True)
b = torch.tensor([[-1], [1], [-1]], dtype=torch.float, requires_grad=True)

c = torch.sum(torch.minimum(a, b))

c.backward()

print(a.grad)
print(b.grad)

