import numpy as np
from .layers.activation import sigmoid, sigmoid_grad, softmax
from .loss import cross_entropy_error, cross_entropy_error_grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化神经网络
        
        参数:
            input_size: 输入层大小（MNIST为784）
            hidden_size: 隐藏层大小
            output_size: 输出层大小（MNIST为10）
            weight_init_std: 权重初始化的标准差
        """
        # 初始化权重和偏置
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """
        预测
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        z1 = np.dot(x, W1) + b1  # 第一层线性变换
        a1 = sigmoid(z1)         # 第一层激活值
        z2 = np.dot(a1, W2) + b2 # 第二层线性变换
        a2 = softmax(z2)         # 第二层激活值（输出）

        return a2

    def loss(self, x, t):
        """
        计算损失
        
        参数:
            x: 输入数据
            t: 真实标签
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        计算准确率
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        """
        计算梯度
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        z1 = np.dot(x, W1) + b1  # 第一层线性变换
        a1 = sigmoid(z1)         # 第一层激活值
        z2 = np.dot(a1, W2) + b2 # 第二层线性变换
        a2 = softmax(z2)         # 第二层激活值（输出）

        # backward
        da2 = (a2 - t) / batch_num      # 输出层的误差
        grads['W2'] = np.dot(a1.T, da2) # 第二层权重的梯度
        grads['b2'] = np.sum(da2, axis=0)

        dz1 = np.dot(da2, W2.T)         # 传递到隐藏层的误差
        da1 = sigmoid_grad(z1) * dz1    # 考虑激活函数的梯度
        grads['W1'] = np.dot(x.T, da1)  # 第一层权重的梯度
        grads['b1'] = np.sum(da1, axis=0)

        return grads

    def train(self, x_train, t_train, epochs=20, batch_size=100, learning_rate=0.1):
        """
        训练网络
        
        参数:
            x_train: 训练数据
            t_train: 训练标签
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
        """
        train_size = x_train.shape[0]
        train_loss_list = []
        train_acc_list = []
        
        for epoch in range(epochs):
            # 打乱数据
            idx = np.random.permutation(train_size)
            x_train = x_train[idx]
            t_train = t_train[idx]
            
            for i in range(0, train_size, batch_size):
                x_batch = x_train[i:i+batch_size]
                t_batch = t_train[i:i+batch_size]
                
                # 计算梯度
                grad = self.gradient(x_batch, t_batch)
                
                # 更新参数
                for key in ('W1', 'b1', 'W2', 'b2'):
                    self.params[key] -= learning_rate * grad[key]
                
            # 记录每个epoch的损失和准确率
            loss = self.loss(x_train, t_train)
            accuracy = self.accuracy(x_train, t_train)
            train_loss_list.append(loss)
            train_acc_list.append(accuracy)
            
            print(f"epoch {epoch + 1} - loss: {loss:.4f}, accuracy: {accuracy:.4f}")
            
        return train_loss_list, train_acc_list
