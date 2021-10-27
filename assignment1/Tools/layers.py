import numpy as np

def layer_forward(x,w,b):
    """
    x: (N,d1,...,dk)  M = d1*...*dk
    w: (D,M)
    b: (M,)
    """
    
    out = None
    
    x_shape = x.shape
    
    num_train = x.shape[0]
    x = x.reshape(num_train,-1)  # (N,D)
    
    out = np.dot(x,w) + b
    
    cache = (x.reshape(x_shape),w,b)
    
    return out, cache
    
    
def layer_backward(dout,cache):
    """
    dout: (N,M)
    
    x: (N,d1,...,dk)
    W: (D,M)
    b: (M,)
    """
    
    x,w,b = cache
    dx,dw,db = None,None,None
    num_train = x.shape[0]
    
    x_shape = x.shape
#     print('x_shape: ',x_shape)
    
    x = x.reshape(num_train,-1)
    
    dx = np.dot(dout,w.T)
    dx = dx.reshape(x_shape)
    
    """
    ???
    dw = np.sum(x.reshape(num_train,-1,1) * dout, axis=0)
    """
    
    dw = np.dot(x.T,dout)
    
    db = np.sum(dout,axis=0)
    
    return dx,dw,db

def relu_forward(x):
    """
    
    """
    
    out = np.maximum(0,x)
    
    cache = x
    return out, cache


def relu_backward(dout,cache):
    dx,x = None,cache
    
    dx = np.zeros(x.shape)
    
    dx[x<0] = 0
    dx[x>=0] = dout[x>=0]
    
    return dx


def layer_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = layer_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def layer_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = layer_backward(da, fc_cache)
    return dx, dw, db

def svm_loss(X,Y):
    """
    X: (N,C)
    Y: (N,)
    """
    
    loss,dX = None,None
    
    N = X.shape[0]
    
    margin = X - X[range(N),Y].reshape(-1,1) + 1
    margin[range(N),Y] = 0
    loss = np.sum(margin[margin>0])
    loss /= N
    
    margin[margin<0] = 0
    margin[margin>0] = 1
    margin[range(N),Y] = -np.sum(margin,axis=1)
    
    #dX = np.sum(margin,axis=0) / N
    dX = margin / N
    
    return loss,dX

def softmax_loss(X,Y):
    """
    X: (N,C)
    Y: (N,)
    """
    N = X.shape[0]
    #X -= np.max(X,axis=1).reshape(-1,1)
    X = np.exp(X)
    exp_sum = np.sum(X,axis=1).reshape(-1,1)
    X = X / exp_sum
    loss = np.sum(-np.log(X[range(N),Y])) / N
    
    
    X[range(N),Y] -= 1
#     dX = X
    dX = X / N
    
    return loss,dX
    
