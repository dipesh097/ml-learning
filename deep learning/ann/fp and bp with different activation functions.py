import numpy as np
import matplotlib.pyplot as plt


x=np.array([[1,0],[0,0],[0,1],[1,1]])
y=np.array([[0],[1],[0],[1]])

w1=np.random.rand(2,2)
b1=np.zeros((1,2))
w2=np.random.rand(2,1)
b2=np.zeros((1,1))

lr=0.1
error_list=[]

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

# for sigmoid function :
for episode in range(600):
    # forward propagation
    z1=np.dot(x,w1)+b1
    a1=sigmoid(z1)
    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)

    error=y-a2
    error_list.append(np.mean(np.abs(error)))
    # backward propagation
    da2=error*sigmoid_derivative(a2)
    da1=np.dot(da2,w2.T)*sigmoid_derivative(a1)

#     upgration of w and b
    w2+=np.dot(a1.T,da2)*lr
    b2+=np.sum(da2,axis=0,keepdims=True)*lr
    w1+=np.dot(x.T,da1)*lr
    b1+=np.sum(da1,axis=0,keepdims=True)*lr
print("final output for sigmoid function is \n",a2.round(3))
plt.plot(error_list)
plt.title("x vs sigmoid function")
plt.show()


# --------------by using tanh function----------------------
error_list1=[]

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1-x**2

w3=np.random.rand(2,2)
b3=np.zeros((1,2))
w4=np.random.rand(2,1)
b4=np.zeros((1,1))

for episode in range(600):
    # forward propagation
    z3=np.dot(x,w3)+b3
    a3=tanh(z3)
    z4=np.dot(a3,w4)+b4
    a4=tanh(z4)
    error1=y-a4
    error_list1.append(np.mean(np.abs(error1)))
    #     backward propagation
    da4=error1*tanh_derivative(a4)
    da3=np.dot(da4,w4.T)*tanh_derivative(a3)

#     upgration of w and b's
    w4+=np.dot(a3.T,da4)*lr
    b4+=np.sum(da4,axis=0,keepdims=True)*lr
    w3+=np.dot(x.T,da3)*lr
    b3+=np.sum(da3,axis=0,keepdims=True)*lr
print("final output by using tanh function is : \n",a4.round(2))
plt.plot(error_list1)
plt.title("by using tanh function")
plt.show()

def relu(x):
    return np.maximum(0,x)
def derivative_relu(x):
    return np.where(x>0,1,0)
w5=np.random.rand(2,2)
b5=np.zeros((1,2))
w6=np.random.rand(2,1)
b6=np.zeros((1,1))

error_list2=[]

for i in range(600):
    z5=np.dot(x,w5)+b5
    a5=relu(z5)
    z6=np.dot(a5,w6)+b6
    a6=relu(z6)

    error2=y-a6
    error_list2.append(np.mean(np.abs(error2)))

    da6=error2*derivative_relu(a6)
    da5=np.dot(da6,w6.T)*derivative_relu(a5)

    w6+=np.dot(a5.T,da6)*lr
    b6+=np.sum(da6,axis=0,keepdims=True)*lr
    w5+=np.dot(x.T,da5)*lr
    b5+=np.sum(da5,axis=0,keepdims=True)*lr
print("final output with relu function :\n",a6.round(3))

plt.plot(error_list2)
plt.title("by using relu function")
plt.grid(True)
plt.show()


















