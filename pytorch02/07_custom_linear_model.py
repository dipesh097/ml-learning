import torch

import torch.nn as nn


class myLinear(nn.Module):

     def __init__(self):
         super(myLinear,self).__init__()
         #  defination parameter

         self.weight=nn.Parameter(torch.randn(1,1))
         self.bias=nn.Parameter(torch.randn(1))

     def forward(self,x):
         return x @ self.weight + self.bias

x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])

y = torch.tensor([[2.0],
                  [4.0],
                  [6.0],
                  [8.0]])

model=myLinear()

loss_fn=nn.MSELoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

for i in range(200):
    optimizer.zero_grad()

    y_pred=model(x)

    loss=loss_fn(y_pred,y)

    loss.backward()

    optimizer.step()

print("finel weight",model.weight)
print("finel bias",model.bias)
