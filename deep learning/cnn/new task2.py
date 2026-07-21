#           input layer ko relu and output layer ko sigmoid se activation
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

def relu(n):
    return np.maximum(0,n)

def derivative_sigmoid(n):
    return n*(1-n)
def relu_derivative(z):
    return np.where(z > 0, 1, 0)


lr = 0.2

for p in range(100000):
    z1=np.dot(x,w1)+b1
    a1=relu(z1)

    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)

    loss=y-a2


#     backward propagation
    da2=loss*derivative_sigmoid(a2)
    da1=np.dot(da2,w2.T)*relu_derivative(a1)

    w2+= np.dot(a1.T,da2)*lr
    b2+=np.sum(da2,axis=0)*lr

    w1+=np.dot(x.T,da1)*lr
    b1+=np.sum(da1,axis=0)*lr
print("outupt :",a2)