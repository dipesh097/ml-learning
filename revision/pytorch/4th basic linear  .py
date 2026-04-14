import torch

x=torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y=torch.tensor([
    [5.0],
    [6.0],
    [7.0],
    [8.0]
    ])

#  module

import torch.nn as nn

model=nn.Linear(in_features=1,out_features=1)

loss_fn=nn.MSELoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

for i in range(50):
    optimizer.zero_grad()

    y_pred=model(x)

    loss=loss_fn(y_pred,y)

    loss.backward()

    optimizer.step()

print("finel  weight",model.weight)
print("finel bias",model.bias)


