# import torch
# import torch.nn as nn
# import torch.nn.functional as f
#
# class simplecnn(nn.Module):
#
#     def __init__(self):
#         super(simplecnn,self).__init__()
#
#         self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3)
#         self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3)
#
#         self.pool=nn.MaxPool2d(2,2)
#         self.fc1=nn.Linear(16*5*5,10)
#
#     def forward(self,x):
#
#             x=self.conv1(x)
#             x=f.relu(x)
#             x=self.pool(x)
#
#             x=self.conv2(x)
#             x=f.relu(x)
#             x=self.pool(x)
#
#             x=torch.flatten(x,1)
#
#             x = self.fc1(x)
#
#             return x
#
# model=simplecnn()
#
# x=torch.randn(1,1,28,28)
#
# output=model(x)
#
# print(output.shape)

#  practice
import torch
import torch.nn as nn
import torch.nn.functional as f

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN,self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3)

        self.max_pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*5*5,10)

    def forward(self,x):
        x=self.conv1(x)
        x=f.relu(x)
        x=self.max_pool(x)

        x=self.conv2(x)
        x=f.relu(x)
        x=self.max_pool(x)

        x=torch.flatten(x)

        x=self.fc1(x)

        return x

x=torch.rand(1,1,26,26)

model=SimpleCNN()

print("input :",x)


print("output:",model(x).shape)