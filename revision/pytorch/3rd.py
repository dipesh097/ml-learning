import torch

x=torch.tensor(4.0,requires_grad=True)

y=x**2+3*x
y.backward()

print(x.grad)