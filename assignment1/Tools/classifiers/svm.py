import numpy as np

def svm_loss_naive(W,X,Y,regu):
    """
    W: (D,C)
    X: (N,D)
    Y: (N,)
    
    return: loss
         d(loss)/dW
    """
    dW = np.zeros(W.shape)
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = np.dot(X[i],W)   # (1,C)
        #print('Y[i]: ',Y)
        correct_class_score = scores[Y[i]]
        for j in range(num_classes):
            if j == Y[i]:
                continue
            else:
                margin = scores[j] - correct_class_score + 1
                if margin>0:
                    loss += margin
                    dW[:,j] += X[i,:]
                    dW[:,Y[i]] -= X[i,:]
    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * regu * np.sum(W*W)
    dW += regu * W
    
    return loss,dW

def svm_loss_vectorized(W,X,Y,regu):
    loss = 0.0
    dW = np.zeros(W.shape)
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    """
    W: (D,C)
    X: (N,D)
    Y: (N,)
    """
    
    scores = np.dot(X,W)  # (N,C)
    correct_class_scores = scores[range(num_train),Y]  # (N,)
    correct_class_scores = np.reshape(correct_class_scores,(num_train,1)) # (N,1)
    
    margin = scores - correct_class_scores + 1   # (N,C)
    margin[range(num_train),Y] = 0.0
    margin[margin<0] = 0.0
    loss += np.sum(margin) / num_train
    loss += 0.5 * regu * np.sum(W*W)
    
    margin[margin>0] = 1
    margin[range(num_train),Y] = - np.sum(margin,axis=1)
    dW = np.dot(X.T,margin)/num_train + regu*W
    return loss,dW
    