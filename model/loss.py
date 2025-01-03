import numpy as np

def cross_entropy_error(y_pred, y_true):
    """
    交叉熵损失函数
    
    参数:
        y_pred: 网络输出
        y_true: 真实标签（one-hot编码）
    """
    if y_pred.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)
        
    # 在计算log时防止出现0
    delta = 1e-7
    return -np.sum(y_true * np.log(y_pred + delta)) / y_pred.shape[0]

def cross_entropy_error_grad(y_pred, y_true):
    """
    交叉熵损失函数的导数
    """
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size
