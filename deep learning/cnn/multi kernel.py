
import numpy as np


image = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])


kernel1=np.array(
                [[1,-1],
                 [-1,1]]
                 )

kernel2=np.array([[0,1],
                 [1,0]]
                 )

def convolution(kernel,image):
    h,w=image.shape
    kh,kw=kernel.shape
    output=np.zeros((h-kh+1,w-kw+1))

    for i in range(h-kh+1):
        for j in range(w-kw+1):
            patch=image[i:i+kh,j:j+kw]
            output[i,j]=np.sum(patch*kernel)
    return output

convo_out1=convolution(kernel1,image)
convo_out2=convolution(kernel2,image)

def relu(x):
    return np.maximum(0,x)

print("output of relu1 is :",relu(convo_out1))

print("output of relu2 is :",relu(convo_out2))



