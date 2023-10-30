import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores) # For numerical stability
        correct_class_score = scores[y[i]]
        exp_sum = np.sum(np.exp(scores)) # Sigma e^scores
        loss += -correct_class_score + np.log(exp_sum)
        for j in range(num_classes):
            softmax_output = np.exp(scores[j]) / exp_sum
            if j == y[i]: # correct class
                dW[:, j] += (softmax_output - 1) * X[i] # softmax -1 because correct class should approach 1 
            else:
                dW[:, j] += softmax_output * X[i] # assign lower probability for non-correct classes

    loss /= num_train
    dW /= num_train

    # Regulization
    if regtype == 'L2':
        loss += reg_l2 * np.sum(W * W)
        dW += 2 * reg_l2 * W
    elif regtype == 'ElasticNet':
        loss += reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))
        dW += 2 * reg_l2 * W + reg_l1 * np.sign(W)

    return loss, dW

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train_num = X.shape[0]

    raw_scores = np.dot(X, W) # raw 
    scores = raw_scores - np.max(raw_scores, axis=1, keepdims=True) # shifting scores again for stabiliy

    exp_scores = np.exp(scores)

    softmax_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_class_probs = softmax_output[np.arange(train_num), y] # number of training X text data matrix 
    
    # vectorize loss += -correct_class_score + np.log(exp_sum)

    loss = -np.sum(np.log(correct_class_probs)) / train_num # where X.shape[0] is num of training (normalization)

    # Regularization
    if regtype == 'L2':
        loss += 0.5 * reg_l2 * np.sum(W * W)
    elif regtype == 'ElasticNet':
        loss += 0.5 * reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))

    # Compute the gradient of the loss with respect to the softmax scores
    dscores = softmax_output
    dscores[np.arange(train_num), y] -= 1 # softmax - 1 in vector form
    dscores /= train_num # Normalization

    # Gradient of loss wrt weights
    dW = np.dot(X.T, dscores)

    # Regularization gradient
    if regtype == 'L2':
        dW += reg_l2 * W
    elif regtype == 'ElasticNet':
        dW += reg_l2 * W + reg_l1 * np.sign(W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
