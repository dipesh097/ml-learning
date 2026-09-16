import torch
import torch.nn as nn
from numpy.ma.core import identity


class ResidualBlock(nn.Module):
    def __init__(self,in_chanel,out_chanel):
        super(ResidualBlock,self).__init__()

        self.conv1=nn.Conv2d(in_chanel,out_chanel,3,padding=1)
        self.bn1=nn.BatchNorm2d(out_chanel)

        self.conv2=nn.Conv2d(out_chanel,out_chanel,3,padding=1)
        self.bn2=nn.BatchNorm2d(out_chanel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = identity + out

        out=torch.relu(x)

        return out


input=torch.randn(1,52,53,53)

block=ResidualBlock(52,52)

y=block(input)

print(y.shape)




