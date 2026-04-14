import numpy as np


image = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

kernel=np.array([[1,-1],
                [-1,1]])

def conv(image,kernel):
    h,w=image.shape
    kh,kw=kernel.shape
    output=np.zeros((h-kh+1,w-kw+1))
    print(output)
    for i in range(h-kh+1):
        for j in range(w-kw+1):
            patch=image[i:i+kh,j:j+kw]
            output[i,j]=np.sum(kernel*patch)
            # print(f"{i}th and {j}th loop, kernel*patch - {kernel*patch} and sum -{np.sum(kernel*patch)}")
            # print(f"patch in {i}th and {j}th loop is in convolution {patch} :")
            # print(f"output in {i}th and {j}th loop is {output} :")
    return output

def relu(x):
    return np.maximum(0,x)


def max_pooling(feature_map, size=2):
    h, w = feature_map.shape
    out_h = h // size
    out_w = w // size

    pooled = np.zeros((out_h, out_w))
    # print("pooled in max pooling:",pooled)

    for i in range(out_h):
        for j in range(out_w):
            pooled[i, j] = np.max(
                feature_map[
                    i*size : i*size + size,
                    j*size : j*size + size
                ]
            )
            # print(f"in {i}th loop pooled is {pooled}")
    return pooled



def flatten(x):
    return x.reshape(-1,1)

def dense(x,w,b):
    return np.dot(w,x)+b

def sigmoid(x):
    return 1/(1+np.exp(-x))

conv_out=conv(image,kernel)
print("final output of conv is :",conv_out)
relu_out=relu(conv_out)
print("relu output is :",relu_out)
pool_out=max_pooling(relu_out)
print("pool output is :",pool_out)
flat_out=flatten(pool_out)
print("output of flatt is :",flat_out)

w_dense=np.random.rand(1,flat_out.shape[0])
b_danse=np.random.rand(1,1)



dense_out=dense(flat_out,w_dense,b_danse)

output=sigmoid(dense_out)

print("final output is: ",output)
