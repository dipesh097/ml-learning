import torch

a=torch.tensor([18,95,86,87,85])

b=torch.mean(a,dtype=float)
print(b)

c=torch.max(a)
print(c)

d=torch.min(a)
print(d)

print(a.shape)

# for print percentage

percentage=(a/100)

print(f"percentage of every student of a tensor is {a/100}% ")

#print index of max and min values









