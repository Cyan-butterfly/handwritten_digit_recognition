import numpy as np
import matplotlib.pyplot as plt

def plot_loss_accuracy(train_loss_list, train_acc_list, save_path=None):
    """
    绘制损失和准确率曲线
    
    参数:
        train_loss_list: 训练损失列表
        train_acc_list: 训练准确率列表
        save_path: 保存图片的路径
    """
    epochs = len(train_loss_list)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制损失曲线
    ax1.plot(range(epochs), train_loss_list, label='train loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(range(epochs), train_acc_list, label='train accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    计算混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
    """
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
    return matrix
