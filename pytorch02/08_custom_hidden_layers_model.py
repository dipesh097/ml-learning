import torch
import torch.nn as nn
from torch.fx.experimental.meta_tracer import nn_layernorm_override


class myann(nn.Module):

    def __init__(self):
        super(myann,self).__init__()

        self.hidden=nn.Linear(1,10)
        self.output=nn.Linear(10,1)

    def forward(self,x):

        x=self.hidden(x)
        x=torch.relu(x)
        x=self.output(x)
        return x

model=myann()

x=torch.linspace(-5,5,100).view(-1,1)
y=x**2

loss_fn=nn.MSELoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.1)

for i in range(1000):
    optimizer.zero_grad()

    y_pred=model(x)

    loss=loss_fn(y_pred,y)

    loss.backward()

    optimizer.step()

print(model(torch.tensor([[3.0]])))


