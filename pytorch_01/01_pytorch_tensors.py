import torch

print(torch.tensor(5))
print(torch.rand((4,5)))
a = torch.zeros((2,3))
print(a)

print(torch.ones((2,2)))

c=torch.tensor([[.1,.23,.23],
               [.3,34,32]])

print(c.dtype)

print(c.device)

model=c.cuda()
print(model)
