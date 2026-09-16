import torch
import torch.nn.functional as f

image=torch.tensor([[1,2,3],
                   [4,5,6],
                   [4,7,8]])

kernel=torch.tensor([[1,-1],
                    [-1,1]])


output=f.conv2d(image.unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0))
print(f"the output is :{output}")
print(f"the output of image.unsqueeze(0).unsqueeze(0)is {image.unsqueeze(0).unsqueeze(0)}")
print(f"the output of kernel.unsqueeze(0) is {kernel.unsqueeze(0)}")
# print(image.unsqueeze(0).unsqueeze(0).shape)
# print(image.shape)
# print(image.unsqueeze(0).unsqueeze(0).squeeze(0).squeeze(0).shape)




