import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from model.network import TwoLayerNet
from dataset.mnist import load_mnist

def train_network(hidden_size, learning_rate, iters_num=2000, batch_size=100):
    """训练网络并返回训练历史
    
    Args:
        hidden_size (int): 隐藏层大小
        learning_rate (float): 学习率
        iters_num (int): 迭代次数
        batch_size (int): 批量大小
    
    Returns:
        dict: 包含训练历史的字典
    """
    # 加载MNIST数据集
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    # 初始化网络
    network = TwoLayerNet(input_size=784, hidden_size=hidden_size, output_size=10)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'parameters': {
            'hidden_size': hidden_size,
            'learning_rate': learning_rate
        }
    }
    
    # 记录训练用时
    train_size = x_train.shape[0]
    
    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 计算梯度
        grad = network.gradient(x_batch, t_batch)
        
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
            
        # 记录训练过程
        if i % 100 == 0:
            # 计算训练损失
            loss = network.loss(x_batch, t_batch)
            history['train_loss'].append(loss)
            
            # 计算训练准确率
            train_acc = network.accuracy(x_train, t_train)
            history['train_acc'].append(train_acc)
            
            # 计算测试准确率
            test_acc = network.accuracy(x_test, t_test)
            history['test_acc'].append(test_acc)
            
            print(f"iter {i}: loss={loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    return history

def plot_training_history(histories, save_dir='results'):
    """绘制训练历史
    
    Args:
        histories (list): 训练历史列表
        save_dir (str): 保存图像的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Iterations (x100)')
    ax1.set_ylabel('Loss')
    
    # 绘制准确率曲线
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Iterations (x100)')
    ax2.set_ylabel('Accuracy')
    
    # 为每个实验绘制曲线
    for history in histories:
        params = history['parameters']
        label = f"hidden={params['hidden_size']}, lr={params['learning_rate']}"
        
        # 绘制损失曲线
        ax1.plot(history['train_loss'], label=label)
        
        # 绘制准确率曲线
        ax2.plot(history['train_acc'], label=label+' (train)')
        ax2.plot(history['test_acc'], label=label+' (test)', linestyle='--')
    
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def main():
    """运行实验"""
    # 实验配置
    experiments = [
        {'hidden_size': 25, 'learning_rate': 0.1},
        {'hidden_size': 50, 'learning_rate': 0.1},
        {'hidden_size': 100, 'learning_rate': 0.1},
        {'hidden_size': 50, 'learning_rate': 0.01},
        {'hidden_size': 50, 'learning_rate': 0.5},
    ]
    
    # 运行实验
    histories = []
    for exp in experiments:
        print(f"\nRunning experiment with hidden_size={exp['hidden_size']}, learning_rate={exp['learning_rate']}")
        history = train_network(hidden_size=exp['hidden_size'], 
                              learning_rate=exp['learning_rate'])
        histories.append(history)
    
    # 绘制结果
    plot_training_history(histories)

if __name__ == '__main__':
    main()
