from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) 

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train

    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)

    Y_hat = X @ W
    N = len(y)
    
    y_hat_true = Y_hat[range(N), y][:, np.newaxis]
    margins = np.maximum(Y_hat - y_hat_true + 1, 0)
    loss = margins.sum() / N - 1 + reg + np.sum(W**2)

    dW = (margins > 0).astype(int)    
    dW[range(N), y] -= dW.sum(axis=1) 
    dW = X.T @ dW / N + 2 * reg * W   

    return loss, dW