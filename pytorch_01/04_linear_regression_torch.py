import torch


x=torch.tensor([1.0,2.,3.,4.,5.,6.])
y_true=x**2

w=torch.tensor(0.,requires_grad=True)

y_pred=x*w

loss=((y_pred-y_true)**2).mean()

loss.backward()
print(w.grad)

