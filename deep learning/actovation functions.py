import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-10,10,50)

sigmoid=1/(1+np.exp(-x))
tanh=np.exp(x)-np.exp(-x)/(np.exp(x))+np.exp(-x)
relu=np.maximum(0,x)
leakey_relu=np.where(x>0,x,x*0.001)
#
# plt.figure(figsize=(10,6))
plt.plot(x,sigmoid)
plt.ylabel("sigmoid")
plt.show()
plt.plot(x,tanh)
plt.ylabel("tanh")
plt.show()
plt.plot(x,relu)
plt.ylabel("relu")
plt.show()
plt.plot(x,leakey_relu)
plt.ylabel("leakey_relu")

plt.title("x cs activation functions")
plt.show()


