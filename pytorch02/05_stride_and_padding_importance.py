import torch
import torch.nn as nn

image=torch.randn(1,1,5,5)

conv1=nn.Conv2d(1,1,kernel_size=3,stride=1)
conv2d=nn.Conv2d(1,1,3,2)

conv3d=nn.Conv2d(1,1,3)
conv4d=nn.Conv2d(1,1,3,padding=1)

out_1=conv1(image)
out_2=conv2d(image)
out_3=conv3d(image)
out_4=conv4d(image)

print("with stride-1 conv_out is ",out_1.shape)
print("with stride-2 conv_out is ",out_2.shape)
print("with spadding-0 conv_out is ",out_3.shape)
print("with spadding-1 conv_out is ",out_4.shape)