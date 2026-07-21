import torch
import torch.nn as nn
import torch.nn.functional as f



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3)
        self.max_pool=nn.MaxPool2d(3,3)
        self.fc1=nn.Linear(64,10)

    def forward(self,x):
        x=self.conv1(x)
        x=f.relu(x)
        x=self.max_pool(x)

        x=self.conv2(x)
        x=f.relu(x)
        x=self.max_pool(x)

        x=torch.flatten(x,1)

        x=self.fc1(x)

        return x

x=torch.randn(32,1,26,26)
y=torch.randint(0,10,(32,))

model=SimpleCNN()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

loss_fn=nn.CrossEntropyLoss()

# farward pass

output=model(x)

loss=loss_fn(output,y)

# backward
optimizer.zero_grad()

loss.backward()

optimizer.step()

print("finel output ",output.shape)






