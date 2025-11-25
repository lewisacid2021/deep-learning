# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    # 数值稳定版
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])

    out = np.zeros_like(x, dtype=float)
    out[pos_mask] = 1 / (1 + z[pos_mask])
    out[neg_mask] = z[neg_mask] / (1 + z[neg_mask])

    return out


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad

    

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-15)
    
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-15)



def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # 最大避免 log(0)
    return -np.sum(np.log(np.maximum(y[np.arange(batch_size), t], 1e-15))) / batch_size



def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
