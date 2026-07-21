import torch

a=torch.tensor(2.0,requires_grad=True)

y=a**4

y.backward()
print(a.grad)