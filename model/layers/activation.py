import numpy as np

def sigmoid(x):
    """
    Sigmoid激活函数
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    """
    Sigmoid函数的导数
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    """
    Softmax激活函数
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
