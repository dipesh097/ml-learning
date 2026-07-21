

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# transform

transform=transforms.Compose([transforms.ToTensor()])

# load dataset

train_dataset=torchvision.datasets.MNIST(

      root='./data',
      train=True,
    download=True,
    transform=transform
    )


train_loader=torch.utils.data.DataLoader(
       train_dataset,
    batch_size=64,
       shuffle=True
)

# define cnn model

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.Conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding=1)
        self.Conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=1)
        self.max_pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*7*7,10)

    def forward(self,x):
        x=(self.max_pool(torch.relu(self.Conv1(x))))
        x=(self.max_pool(torch.relu(self.Conv2(x))))
        x=torch.flatten(x,1)
        x=self.fc1(x)

        return x

# set up training

model=CNN()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# traing loop

for i in range(3):
    for images,labels in train_loader:

        optimizer.zero_grad()

        output=model(images)

        loss=loss_fn(output,labels)

        loss.backward()

        optimizer.step()


# accuracy check

correct=0
total=0

with torch.no_grad():
    for images , labels in train_loader:

        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total=total+labels.size(0)
        correct = correct+(predicted==labels).sum().item()
print("accuracy",100*(correct/total))




