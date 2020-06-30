from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    f=X.dot(W)
    num_train=X.shape[0]
    classes=W.shape[1]

    for i in range(num_train):
        scores=f[i]-np.max(f[i])
        correct_score=scores[y[i]]
        denominator=np.sum(np.exp(scores))
        probabilities=np.exp(scores)/denominator
        loss+=np.log(denominator)-correct_score
        dW[:,y[i]]-=X[i]
        for j in range(classes):
            dW[:,j]+=X[i]*probabilities[j]
    #average
    loss/=num_train
    dW/=num_train
    #regularization
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    f=X.dot(W)
    #loss
    #make the scores 'stable'
    highest=f[np.arange(num_train),f.argmax(axis=1)]
    stable=(f.T-highest).T
    #calculate loss
    denominator=np.sum(np.exp(stable),axis=1)
    numerator=np.exp(stable[np.arange(num_train),y])
    losses=-np.log(numerator/denominator)
    loss=np.sum(losses)/num_train
    #regularize
    loss+=reg*np.sum(W*W)

    #gradient
    coeffs=(np.exp(stable).T/denominator).T
    stable[:,:]=0
    stable[np.arange(num_train),y]=1
    coeffs-=stable
    dW=(X.T).dot(coeffs)
    #average
    dW/=num_train
    #regularize
    dW+=2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
