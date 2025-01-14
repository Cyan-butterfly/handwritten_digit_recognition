import numpy as np

def sigmoid(x):
    """
    Sigmoid激活函数
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(out):
    """
    Sigmoid函数的导数
    注意：这里的参数是sigmoid函数的输出值，而不是输入值
    """
    return out * (1.0 - out)

def softmax(x):
    """
    Softmax激活函数
    
    Args:
        x: 输入数组，shape (N, C) 其中 N 是样本数，C 是类别数
        
    Returns:
        输出数组，shape与输入相同
    """
    x = x - np.max(x, axis=1, keepdims=True)  # 为了数值稳定性，减去最大值
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    """
    ReLU激活函数
    """
    return np.maximum(0, x)

def relu_grad(x):
    """
    ReLU函数的导数
    """
    return (x > 0).astype(np.float32)
