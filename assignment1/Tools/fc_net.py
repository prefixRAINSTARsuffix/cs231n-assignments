import numpy as np

from .layers import *

class TwoLayerNet(object):
    
    def __init__(
        self,
        input_dim = 3*32*32,
        hidden_dim = 100,
        num_classes = 10,
        weight_scale=1e-3,
        reg=0.0
    ):
        
        self.params = {}
        self.reg = reg
        
        self.params['W1'] = np.random.randn(input_dim,hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        
        self.params['W2'] = np.random.randn(hidden_dim,num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
        
    
    def loss(self,X,Y=None):
        
        """
        X: (N,d1,...,dk)
        Y: (N,)
        affine - relu - affine - softmax
        """
        
        scores,cache1 = layer_forward(X,self.params['W1'],self.params['b1'])
        scores,cache_relu = relu_forward(scores)
        scores,cache2 = layer_forward(scores,self.params['W2'],self.params['b2'])
        
        if Y is None:
            return scores
        
        loss,grads = 0.0,{}
        
        N = X.shape[0]
        loss,dout = softmax_loss(scores,Y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1'])
                              + np.sum(self.params['W2']*self.params['W2']))
        
        dx,dw,db = layer_backward(dout,cache2)
        grads['W2'],grads['b2'] = dw + self.reg * self.params['W2'],db
        dx = relu_backward(dx,cache_relu)
        _,dw,db = layer_backward(dx,cache1)
        grads['W1'],grads['b1'] = dw + self.reg * self.params['W1'],db
        
        return loss,grads
        
        