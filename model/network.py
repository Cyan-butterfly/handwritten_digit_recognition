import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.layers.activation import relu, relu_grad, softmax, sigmoid,sigmoid_grad
from model.loss import cross_entropy_error, cross_entropy_error_grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        self.params['b2'] = np.zeros(output_size)
        self.cache = {}

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # 前向传播
        z1 = np.dot(x, W1) + b1
        # a1 = sigmoid(z1)
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y = softmax(z2)
        
        # 保存中间值
        self.cache = {
            'x': x,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'y': y
        }
        
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 前向传播已经在loss()中完成，直接使用cache中的值
        x = self.cache['x']
        a1 = self.cache['a1']
        z1 = self.cache['z1']
        y = self.cache['y']
        
        grads = {}
        batch_size = x.shape[0]
        
        # 反向传播
        # 输出层
        dy = (y - t) / batch_size
        grads['W2'] = np.dot(a1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        # 隐藏层
        da1 = np.dot(dy, self.params['W2'].T)
        dz1 = relu_grad(z1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads

    def train(self, x_train, t_train, learning_rate=0.1, batch_size=100, epochs=10):
        train_size = x_train.shape[0]
        iterations_per_epoch = max(train_size // batch_size, 1)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            for i in range(iterations_per_epoch):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]
                
                # 先计算loss，这会进行前向传播并更新cache
                loss = self.loss(x_batch, t_batch)
                # 计算梯度
                grads = self.gradient(x_batch, t_batch)
                
                # 更新参数
                for key in ('W1', 'b1', 'W2', 'b2'):
                    self.params[key] -= learning_rate * grads[key]
                
                if i % 100 == 0:
                    loss = self.loss(x_batch, t_batch)
                    print(f"iteration {i}: loss = {loss:.4f}")

class Network:
    def __init__(self, hidden_size=50, learning_rate=0.1):
        self.input_size = 784  # 28x28
        self.hidden_size = hidden_size
        self.output_size = 10
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
    def train(self, X, y, epochs=10, batch_size=32, logger=None):
        """
        训练网络
        
        Args:
            X: 输入数据
            y: 标签
            epochs: 训练轮数
            batch_size: 批次大小
            logger: 日志记录器
        """
        history = {'loss': [], 'accuracy': []}
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            epoch_loss = 0
            correct_predictions = 0
            
            # 批次训练
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # 前向传播
                output = self.forward(batch_X)
                loss = self.compute_loss(output, batch_y)
                epoch_loss += loss
                
                # 计算准确率
                predictions = np.argmax(output, axis=1)
                correct_predictions += np.sum(predictions == batch_y)
                
                # 反向传播
                self.backward(batch_X, batch_y, output)
            
            # 计算epoch平均损失和准确率
            epoch_loss /= (n_samples // batch_size)
            epoch_accuracy = correct_predictions / n_samples
            
            # 记录历史
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            
            # 使用logger记录每个epoch的信息
            if logger:
                logger.log_epoch(epoch, {
                    'loss': epoch_loss,
                    'accuracy': epoch_accuracy
                })
        
        return history

    def backward(self, X, y, output):
        # 计算输出层梯度
        output_error = output.copy()
        output_error[range(len(y)), y] -= 1
        
        # 计算隐藏层的输出
        hidden = np.dot(X, self.W1) + self.b1
        hidden = self.sigmoid(hidden)
        
        # 更新输出层权重
        self.W2 -= self.learning_rate * np.dot(hidden.T, output_error) / len(y)
        self.b2 -= self.learning_rate * np.mean(output_error, axis=0)
        
        # 计算隐藏层梯度
        hidden_grad = np.dot(output_error, self.W2.T) * (hidden * (1 - hidden))  # sigmoid导数
        
        # 更新隐藏层权重
        self.W1 -= self.learning_rate * np.dot(X.T, hidden_grad) / len(y)
        self.b1 -= self.learning_rate * np.mean(hidden_grad, axis=0)
    def forward(self, X):
        # 输入层到隐藏层
        hidden = np.dot(X, self.W1) + self.b1
        hidden = self.sigmoid(hidden)
        
        # 隐藏层到输出层
        output = np.dot(hidden, self.W2) + self.b2
        output = self.softmax(output)
        
        return output  # 只返回输出层结果
    
    def softmax(self, x):
        # 数值稳定性处理
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def compute_loss(self, output, y):
        # 交叉熵损失
        m = y.shape[0]
        log_likelihood = -np.log(output[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))