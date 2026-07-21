# # this is for xor gate
# import numpy as np
# import random
#
# x=np.array([[1,0],
#            [1,1],
#            [0,0],
#            [0,1]])
#
# y=np.array([[1],[0],[0],[1]])
#
# w1=np.random.rand(2,2)
# b1=np.zeros((1,2))
# w2=np.random.rand(2,1)
# b2=np.zeros((1,1))
#
#
# lr=0.2
#
# def sigmoid(n):
#     return 1/(1+np.exp(-n))
# def sigmoid_derivative(n):
#     return n*(1-n)
#
# for i in range(50000):
#     z1=np.dot(x,w1)+b1
#     a1=sigmoid(z1)
#     z2=np.dot(a1,w2)+b2
#     a2=sigmoid(z2)
#
#
#     error=y-a2
#     da2=error*sigmoid_derivative(a2)
#     da1=np.dot(da2,w2.T)*sigmoid_derivative(a1)
#
#     w2+=np.dot(a1.T,da2)*lr
#     b2+=np.sum(da2,axis=0,keepdims=True)*lr
#     w1+=np.dot(x.T,da1)*lr
#     b1+=np.sum(da1,axis=0,keepdims=True)*lr
#     # print(f"weight1 for {i}th is {w1}")
#     # print(f"weight2 for {i}th is {w2}")
#     # print(f"bias1 for {i}th is {b1}")
#     # print(f"bias2 for {i}th is {b1}")
#
# print("output after training :")
# print(a2.round(4))


import numpy as np
import random

x=np.array([[1,0],
           [1,1],
           [0,0],
           [0,1]])

w1=np.random.rand(2,2)
b1=np.zeros((1,2))

y=np.array([[1],[0],[0],[1]])

w2=np.random.rand(2,1)
b2=np.zeros((1,1))

lr=0.2

def sigmoid(n):
    return 1/(1+np.exp(-n))

def derivative_sigmoid(n):
    return n*(1-n)


for i in range(50000):
    z1=np.dot(x,w1)+b1
    a1=sigmoid(z1)

    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)

    error=y-a2

    da2=error*derivative_sigmoid(a2)

    da1=np.dot(da2,w2.T)*derivative_sigmoid(a1)

    w2+=np.dot(a1.T,da2)*lr
    b2+=np.sum(da2,axis=0)*lr
    w1+=np.dot(x.T,da1)*lr
    b1+=np.sum(da1,axis=0)*lr

print("output (a2):",a2)

