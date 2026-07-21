import torch
import torch.nn as nn

conv1x1 = nn.Conv2d(
    in_channels=3,
    out_channels=2,
    kernel_size=1
)

x = torch.randn(1,3,4,4)

y = conv1x1(x)

print(y.shape)