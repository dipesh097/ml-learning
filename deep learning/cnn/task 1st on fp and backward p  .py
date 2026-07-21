import numpy as np
import random
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

w1=np.random.rand(2,2)
b1=np.zeros((1,2))

w2=np.random.rand(2,1)
b2=np.zeros((1,1))

def sigmoid(n):
    return 1/(1+np.exp(-n))

def derivative_sigmoid(n):
    return n*(1-n)

lr = 0.2

for p in range(100000):
    z1=np.dot(x,w1)+b1
    a1=sigmoid(z1)

    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)

    loss=y-a2


#     backward propagation
    da2=loss*derivative_sigmoid(a2)
    da1=np.dot(da2,w2.T)*derivative_sigmoid(a1)

    w2+= np.dot(a1.T,da2)*lr
    b2+=np.sum(da2,axis=0)*lr

    w1+=np.dot(x.T,da1)*lr
    b1+=np.sum(da1,axis=0)
print("outupt :",a2)
