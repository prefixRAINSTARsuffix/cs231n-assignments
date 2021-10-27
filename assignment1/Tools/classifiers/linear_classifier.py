from ..classifiers.svm import *
from ..classifiers.softmax import *
import numpy as np

class LinearClassifier(object):
    def __init__(self):
        self.W = None
        
    def train(
        self,
        X,
        Y,
        learning_rate=1e-3,
        regu=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False
    ):
        """
        X: (N,D)
        Y: (N,)
        W: (D,C)
        """
        num_train,dim = X.shape
        num_classes = np.max(Y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim,num_classes)
            
        
        # SGD
        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(range(num_train),batch_size)
            X_batch = X[indices]
            Y_batch = Y[indices]
            loss,grad = self.loss(X_batch,Y_batch,regu)
            loss_history.append(loss)
            
            self.W -= learning_rate * grad
            
            if verbose and it%100 == 0:
                print("iteration {} / {} : loss {}".format(it,num_iters,loss))
        
        return loss_history
    
    
    def predict(self,X):
        """
        X: (N,D)
        W: (D,C)
        """
        Y_pred = np.zeros(X.shape[0])
        
        scores = np.dot(X,self.W)  # (N,C)
        Y_pred = np.argmax(scores,axis=1)
        
        return Y_pred
    
    def loss(self,X_batch,Y_batch,regu):
        pass
    
class LinearSVM(LinearClassifier):
    
    def loss(self,X_batch,Y_batch,regu):
        return svm_loss_vectorized(self.W,X_batch,Y_batch,regu)
    
class LinearSoftmax(LinearClassifier):
    
    def loss(self,X_batch,Y_batch,regu):
        return softmax_loss_vectorized(self.W,X_batch,Y_batch,regu)
            