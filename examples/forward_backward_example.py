'''
这是一个示例程序，展示了神经网络的前向传播和反向传播过程。
'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    x = x - np.max(x)  # 防止数值溢出
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

# 1. 初始化数据
x = np.array([0.5, 0.8])          # 输入：2个像素值
t = np.array([1, 0])              # 目标：第一类

# 2. 初始化网络参数
W1 = np.array([[0.1, 0.2],        # 第一层权重（2x2矩阵）
               [0.3, 0.4]])
b1 = np.array([0.1, 0.1])         # 第一层偏置

W2 = np.array([[0.1, 0.3],        # 第二层权重（2x2矩阵）
               [0.2, 0.4]])
b2 = np.array([0.1, 0.1])         # 第二层偏置

print("===== 前向传播 =====")
print("\n输入形状:")
print(f"x shape: {x.shape}")       # 应该是 (2,)
print(f"W1 shape: {W1.shape}")     # 应该是 (2, 2)
print(f"b1 shape: {b1.shape}")     # 应该是 (2,)

# 3. 第一层前向传播
print("\n第一层：")
z1 = np.dot(x, W1) + b1           # 线性变换
print(f"z1 = np.dot(x, W1) + b1 = {z1}")
print(f"z1 shape: {z1.shape}")     # 应该是 (2,)

a1 = sigmoid(z1)                   # 激活函数
print(f"a1 = sigmoid(z1) = {a1}")
print(f"a1 shape: {a1.shape}")     # 应该是 (2,)

# 4. 第二层前向传播
print("\n第二层：")
z2 = np.dot(a1, W2) + b2          # 线性变换
print(f"z2 = np.dot(a1, W2) + b2 = {z2}")
print(f"z2 shape: {z2.shape}")     # 应该是 (2,)

a2 = softmax(z2)                   # 输出层激活
print(f"a2 = softmax(z2) = {a2}")
print(f"a2 shape: {a2.shape}")     # 应该是 (2,)

# 5. 计算损失
loss = cross_entropy_error(a2, t)
print(f"\n损失值 = {loss}")

print("\n===== 反向传播 =====")
# 6. 输出层的反向传播
print("\n输出层梯度：")
da2 = (a2 - t)                     # 输出层误差（sigmoid + softmax简化了输出层的梯度计算）
print(f"da2 = (a2 - t) = {da2}")
print(f"da2 shape: {da2.shape}")   # 应该是 (2,)

# 重塑维度以进行矩阵乘法
a1_reshaped = a1.reshape(-1, 1)    # 变成 (2, 1)
da2_reshaped = da2.reshape(1, -1)  # 变成 (1, 2)

dW2 = np.dot(a1_reshaped, da2_reshaped)  # 第二层权重梯度 （dw2 = da2 * a1.T）
print(f"dW2 = np.dot(a1.reshape(-1,1), da2.reshape(1,-1)) =\n{dW2}")
print(f"dW2 shape: {dW2.shape}")   # 应该是 (2, 2)

db2 = da2                          # 第二层偏置梯度
print(f"db2 = da2 = {db2}")


# 7. 隐藏层的反向传播
print("\n隐藏层梯度：")
da1 = np.dot(da2, W2.T)           # 传递到隐藏层的误差
print(f"da1 = np.dot(da2, W2.T) = {da1}")

dz1 = sigmoid_grad(z1) * da1      # 考虑激活函数的梯度
print(f"dz1 = sigmoid_grad(z1) * da1 = {dz1}")

# 重塑维度以进行矩阵乘法
x_reshaped = x.reshape(-1, 1)      # 变成 (2, 1)
dz1_reshaped = dz1.reshape(1, -1)  # 变成 (1, 2)

dW1 = np.dot(x_reshaped, dz1_reshaped)  # 第一层权重梯度
print(f"dW1 = np.dot(x.reshape(-1,1), dz1.reshape(1,-1)) =\n{dW1}")
print(f"dW1 shape: {dW1.shape}")   # 应该是 (2, 2)
db1 = dz1                          # 第一层偏置梯度
print(f"db1 = dz1 = {db1}")

# 8. 参数更新示例
learning_rate = 0.1
print("\n===== 参数更新 =====")
print(f"学习率 = {learning_rate}")

W1_new = W1 - learning_rate * dW1
print(f"\n更新后的W1 =\n{W1_new}")

W2_new = W2 - learning_rate * dW2
print(f"\n更新后的W2 =\n{W2_new}")
