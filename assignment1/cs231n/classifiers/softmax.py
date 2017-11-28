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

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        f = X[i].dot(W)
        f -= np.max(f)    # numeric stability
        s = np.sum(np.exp(f))
        loss += np.log(s) - f[y[i]]
        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += np.exp(f[j]) / s * X[i]
            else:
                dW[:, y[i]] += (-1 + np.exp(f[j]) / s) * X[i]
    loss = loss / num_train
    dW = dW / num_train
    loss += 0.5 * reg * np.sum(np.square(W))
    dW += reg * W
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    num_train = X.shape[0]
    f = np.dot(X, W)  # [N, C]
    f -= f.max(axis=1).reshape(num_train, 1)
    s = np.sum(np.exp(f), axis=1)   # [N,]
    data_loss = np.sum(np.log(s)) - np.sum(f[range(num_train), y])
    df = np.exp(f) / s.reshape(num_train, 1)
    df[range(num_train), y] -= 1
    dW = np.dot(X.T, df)

    data_loss = data_loss / num_train
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    dW = dW / num_train + reg * W

    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW
