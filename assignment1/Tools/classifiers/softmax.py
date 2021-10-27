import numpy as np

def softmax_loss_naive(W,X,Y,regu):
    """
    W: (D,C)
    X: (N,D)
    Y: (N,)
    """
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        scores = np.dot(X[i],W) # (C,)
        scores -= np.max(scores)
        sum_exp = np.sum(np.exp(scores))
        loss += -np.log(np.exp(scores[Y[i]]) / sum_exp)
        
        for j in range(num_classes):
            if j==Y[i]:
                dW[:,j] += -X[i]
            else:
                dW[:,j] += np.exp(scores[j]) * X[i] / sum_exp
                
    loss /= num_train
    loss += 0.5 * regu * np.sum(W * W)
    
    dW /= num_train
    dW += regu * W
    
    return loss, dW


def softmax_loss_vectorized(W,X,Y,regu):
    """
    W: (D,C)
    X: (N,D)
    Y: (N,)
    """
    
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = np.dot(X,W)  # (N,C)
    scores -= np.max(scores,axis=1).reshape(-1,1)
    scores = np.exp(scores)
    sum_exps = np.sum(scores,axis=1).reshape(-1,1) # (N,1)
    
    """
    why different?
    loss += np.sum(-np.log(scores[range(num_train),Y]/sum_exps))
    """
    scores /= sum_exps
    loss += np.sum(-np.log(scores[range(num_train),Y]))
    
    
    scores[range(num_train),Y] -= 1
    dW = np.dot(X.T,scores)
    
    loss /= num_train
    loss += 0.5 * regu * np.sum(W * W)
    
    dW /= num_train
    dW += regu * W
    
    return loss, dW
    
    