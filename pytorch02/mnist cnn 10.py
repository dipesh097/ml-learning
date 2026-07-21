import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset=torchvision.datasets.MNIST(

    root='./data',
    train=True,
    download=True,
    transform=transform
)

data_loader=torch.utils.data.DataLoader(

    train_dataset,
    batch_size=64,
    shuffle=True
    )


class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()

            self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
            self.conv2=nn.Conv2d(16,32,3,1,1)
            self.max_pool=nn.MaxPool2d(2,2)
            self.fc1=nn.Linear(32*7*7,128)
            self.drop_out=nn.Dropout(0.5)
            self.fc2=nn.Linear(128,10)

        def forward(self,x):
            x=self.max_pool(torch.relu(self.conv1(x)))
            x=self.max_pool(torch.relu(self.conv2(x)))
            x=torch.flatten(x,1)
            x=torch.relu(self.fc1(x))
            x=self.drop_out(x)   #dropout add for improving accuracy and reducing overfiting
            x=self.fc2(x)
            return x

model=CNN()

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epoches=3

for epoch in range(epoches):

    correct=0
    total=0

    for images , labels in data_loader:

        optimizer.zero_grad()

        output=model(images)

        loss=criterion(output,labels)
        # print(f"loss is {loss} \n ")

        loss.backward()

        optimizer.step()

        _,predicted=torch.max(output,1)

        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print("accuracy",100*(correct/total))







