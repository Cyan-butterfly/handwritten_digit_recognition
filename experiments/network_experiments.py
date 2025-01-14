import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import TwoLayerNet

# 1.3生成 实验脚本
# 这个脚本会：

# 训练5个不同配置的网络：
# 不同隐藏层大小（25, 50, 100）保持学习率0.1
# 不同学习率（0.01, 0.1, 0.5）保持隐藏层大小50
# 记录每个网络的：
# 训练损失
# 训练准确率
# 测试准确率
# 生成两张图表：
# 损失函数随时间的变化
# 准确率随时间的变化（包括训练集和测试集）
# 运行后，我们可以分析不同配置的效果，看看：

# 隐藏层大小如何影响模型性能
# 学习率如何影响训练速度和稳定性
# 是否存在过拟合现象
def train_network(hidden_size=100, learning_rate=0.1, batch_size=100, iters_num=1000):
    """
    训练网络
    """
    # 加载数据
    (x_train, t_train), (x_test, t_test) = load_mnist_data()
    
    train_size = x_train.shape[0]
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    # 打印数据加载详情
    print("\n=== 数据加载详情 ===")
    print(f"x_train: shape={x_train.shape}, dtype={x_train.dtype}, min={x_train.min()}, max={x_train.max()}")
    print(f"x_test: shape={x_test.shape}, dtype={x_test.dtype}, min={x_test.min()}, max={x_test.max()}")
    print(f"y_train: shape={t_train.shape}, dtype={t_train.dtype}, unique values={np.unique(t_train)}")
    print(f"y_test: shape={t_test.shape}, dtype={t_test.dtype}, unique values={np.unique(t_test)}")
    
    if len(t_train.shape) == 2:
        print("\n标签已经是one-hot编码")
    else:
        print("\n将标签转换为one-hot编码")
        t_train = to_one_hot(t_train)
        t_test = to_one_hot(t_test)
    
    print("\n=== 最终数据形状 ===")
    print(f"x_train: {x_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_train: {t_train.shape}")
    print(f"y_test: {t_test.shape}")
    
    # 初始化网络
    network = TwoLayerNet(input_size=784, hidden_size=hidden_size, output_size=10)
    
    print("\n训练参数:")
    print(f"隐藏层大小: {hidden_size}")
    print(f"学习率: {learning_rate}")
    print(f"批量大小: {batch_size}")
    print(f"迭代次数: {iters_num}")
    
    # 打印初始预测
    print("\n初始预测示例:")
    initial_pred = network.predict(x_train[:5])
    print("预测概率分布:")
    print(initial_pred)
    print("真实标签:")
    print(t_train[:5])
    
    # 训练
    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 计算梯度
        loss = network.loss(x_batch, t_batch)  # 这会更新cache
        grads = network.gradient(x_batch, t_batch)
        
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grads[key]
        
        # 记录学习过程
        if i % 100 == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_loss_list.append(loss)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"iter {i}: loss={loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
            
            if i == iters_num // 2:
                print("\n训练中预测示例:")
                mid_pred = network.predict(x_train[:5])
                print("预测概率分布:")
                print(mid_pred)
                print("真实标签:")
                print(t_train[:5])
    
    return {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_acc': test_acc_list
    }

def load_mnist_data():
    """加载MNIST数据集"""
    print("\n=== 开始训练网络 ===")
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 将图像数据展平并归一化到[0,1]
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # 将标签转换为one-hot编码
    y_train = keras.utils.to_categorical(y_train, 10).astype('float32')
    y_test = keras.utils.to_categorical(y_test, 10).astype('float32')
    
    return (x_train, y_train), (x_test, y_test)

def to_one_hot(y, num_classes=10):
    y = y.reshape(-1)  # 确保是一维的
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

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
    """主函数"""
    # 实验配置
    experiments = [
        {
            'hidden_size': 256,  # 增大隐藏层
            'learning_rate': 0.1,  # 增大学习率
            'batch_size': 128,  # 增大batch size
            'epochs': 5  # 增加训练轮数
        }
    ]
    
    # 运行实验
    for exp in experiments:
        print(f"\n开始实验: {exp}")
        history = train_network(
            hidden_size=exp['hidden_size'],
            learning_rate=exp['learning_rate'],
            batch_size=exp['batch_size'],
            iters_num=exp['epochs']*600
        )
        plot_training_history([{'parameters': exp, 'train_loss': history['train_loss'], 'train_acc': history['train_acc'], 'test_acc': history['test_acc']}])

if __name__ == '__main__':
    main()
