
import torch

x=torch.tensor([1.,2.])
y=4*x

w=torch.tensor(1.,requires_grad=True)

y_pred=w*x

loss=((y_pred-y)**2).mean()

loss.backward()

print(w.grad)