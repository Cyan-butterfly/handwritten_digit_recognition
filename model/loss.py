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
    batch_size = y_pred.shape[0]
    # 对每个样本分别计算交叉熵，然后取平均
    return -np.sum(y_true * np.log(y_pred + delta)) / batch_size

def cross_entropy_error_grad(y_pred, y_true):
    """
    交叉熵损失函数的导数
    
    注意：这里不需要除以batch_size，因为在损失函数中已经除过了
    """
    return y_pred - y_true
