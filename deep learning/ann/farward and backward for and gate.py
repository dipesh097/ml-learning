# Input (x₁, x₂)	Output (y)
# [0, 0]	0
# [0, 1]	0
# [1, 0]	0
# [1, 1]	1

import numpy as np

import matplotlib.pyplot as plt
x=np.array([[1,0],[0,0],[0,1],[1,1]])
y=np.array([[0],[0],[0],[1]])



w1=np.random.rand(2,2)
b1=np.zeros((1,2))
w2=np.random.rand(2,1)
b2=np.zeros((1,1))

lr=0.1

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

error_list=[]

for i in range(50000):
    z1=np.dot(x,w1)+b1
    a1=sigmoid(z1)

    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)

    error=y-a2
    error_list.append(np.mean(np.abs(error)))

    da2=error*sigmoid_derivative(a2)
    da1=np.dot(da2,w2.T)*sigmoid_derivative(a1)

    w2+=np.dot(a1.T,da2)*lr
    b2+=np.sum(da2,axis=0,keepdims=True)*lr
    w1+=np.dot(x.T,da1)*lr
    b1+=np.sum(da1,axis=0,keepdims=True)*lr
    # print(f" for {i}th itration output i ",a2)


print("final output is ",  a2.round(3))

plt.plot(error_list)
plt.legend(["error"])
plt.grid(True)
plt.show()



