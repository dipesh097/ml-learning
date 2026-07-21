import numpy as np

image = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=float)

y_true=np.array([[1.0]])

kernel=np.random.randn(2,2)*0.1
w_dense=np.random.randn(1,1)*0.1
b_dense=np.zeros((1,1))

lr=0.1


def conv(image,kernel):
    h,w=image.shape
    kh,kw=kernel.shape
    output=np.zeros((h-kh+1,w-kw+1))

    for i in range(h-kh+1):
        for j in range(w-kw+1):
            patch=image[i:i+kh,j:j+kw]
            output[i,j]=np.sum(patch*kernel)
    return output

def relu(x):
    return np.maximum(0,x)

def max_pooling(x,size=2):
    h,w=x.shape
    out_h=h//size
    out_w=w//size

    pool=np.zeros((out_h,out_w))
    mask=np.zeros_like(x)

    for i in range(out_h):
        for j in range(out_w):
            window=x[i*size:i*size+size,j*size:j*size+size]
            m=np.max(window)
            pool[i,j]=m

            mask[i*size:i*size+size,j*size:j*size+size]=(window==m)
    return mask , pool

def flatten(x):
    return x.reshape(-1,1)

def dense(w,x,b):
    return np.dot(x,w)+b

# def sigmoid(x):
#     return 1/(1+np.exp(-1))


# backward fuctions

def relu_backward(dout,x):
    return dout*(x>0)

def back_max(dout,mask,size=2):
    dx=np.zeros_like(mask)
    h,w=dout.shape
    for i in range(h):
        for j in range(w):
            dx[i*size:i*size+size,j*size:j*size+size] +=dout[i,j]*mask[i*size:i*size+size,j*size:j*size+size]

    return dx

def back_convolution(dout,image,kernel):
    kh,kw=kernel.shape
    dkernel=np.zeros_like(kernel)

    for i in range(dout.shape[0]):
        for j in range(dout.shape[1]):
            dkernel+=dout[i,j]*image[i:i+kh,j:j+kw]
    return dkernel

#  training

for i in range(20):
    conv_out=conv(image,kernel)
    relu_out=relu(conv_out)
    pool_mask,pool_out=max_pooling(relu_out)
    flatt=flatten(pool_out)
    output=dense(w_dense,flatt,b_dense)
    # sigmoid_out=sigmoid(dense_out)

    loss=(y_true-output)**2

    d_output=-2*(y_true-output)
    dw_dense=np.dot(d_output,flatt.T)
    db_dense=d_output
    d_flate=np.dot(w_dense.T,d_output)

    d_pool=d_flate.reshape(pool_out.shape)
    d_relu=back_max(d_pool,pool_mask)
    d_conv=relu_backward(d_relu,conv_out)
    d_kernel=back_convolution(d_conv,image, kernel)

#     update
    w_dense -=lr*dw_dense
    b_dense -=lr*db_dense
    kernel -=lr*d_kernel

print("finel output is : ",output)




