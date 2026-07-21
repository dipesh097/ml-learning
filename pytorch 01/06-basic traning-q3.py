import torch


a=torch.tensor([1.,2.,3.,4.,5.])
b_true=5*a+4

w=torch.tensor(0.0,requires_grad=True)

lr=0.01

for i in range(5000):
    b_pred=w*a

    loss=((b_pred-b_true)**2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr* w.grad

    w.grad.zero_()

    print(f"w in {i}th loop is {w} and precticted answer is {b_pred}")



