# import numpy as np
#
# x=np.array([1,0])
#
# w1=np.array([[0.3,0.5],
#         [0.3,0.4]])
# b1=np.array([1,1.5])
#
# w2=np.array([[0.6,0.3],
#              [0.1,0.2]])
# b2=np.array([2,1.6])
#
# def sigmoid(y):
#     return 1/1+np.exp(-y)
#
# z1=np.dot(x,w1)+b1
# a1=sigmoid(z1)
#
# z2=np.dot(a1,w2)+b2
# a2=sigmoid(z2)
#
# print("first hidden layer input ",a1)
# print("final hidden layers",a2)

import numpy as np

x=np.array([1,0])

w1=np.array([[0.3,0.5],
        [0.3,0.4]])
b1=np.array([1,1.5])

w2=np.array([[0.6,0.3],
             [0.1,0.2]])
b2=np.array([2,1.6])

def sigmoid(y):
    return 1/(1+np.exp(-y))

z1=np.dot(x,w1)+b1
a1=sigmoid(z1)

z2=np.dot(a1,w2)+b2
a2=sigmoid(z2)


print("first hiddde layer input:",a1)
print("second hiddem layer input:",a2)