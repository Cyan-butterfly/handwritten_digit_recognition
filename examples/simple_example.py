import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 1. 设置输入数据
x = np.array([0.5, 0.8])    # 输入：2个像素值
t = np.array([1, 0])        # 目标：第一类

# 2. 初始化网络参数
W1 = np.array([[0.1, 0.2],  # 第一层权重（2x2矩阵）
               [0.3, 0.4]])
b1 = np.array([0.1, 0.1])   # 第一层偏置

# 3. 计算第一层（输入层->隐藏层）
print("步骤1: 计算线性变换 z1 = np.dot(x, W1) + b1")
print(f"输入 x: {x}")
print(f"权重 W1:\n{W1}")
print(f"偏置 b1: {b1}")

# 3.1 计算 np.dot(x, W1)
dot_product = np.dot(x, W1)
print("\n3.1 矩阵乘法 np.dot(x, W1):")
print(f"[0.5 * 0.1 + 0.8 * 0.3, 0.5 * 0.2 + 0.8 * 0.4]")
print(f"= [{0.5*0.1 + 0.8*0.3}, {0.5*0.2 + 0.8*0.4}]")
print(f"= {dot_product}")

# 3.2 添加偏置得到线性变换结果
z1 = dot_product + b1
print("\n3.2 添加偏置得到z1:")
print(f"{dot_product} + {b1}")
print(f"= {z1}")

# 4. 应用激活函数得到激活值
a1 = sigmoid(z1)
print("\n步骤2: 应用sigmoid激活函数得到a1")
print(f"a1 = sigmoid({z1})")
print(f"= {a1}")
