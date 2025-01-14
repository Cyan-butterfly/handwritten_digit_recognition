'''
这是一个预测单个图像的示例，调用了predict_digit和visualize_prediction函数
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from model.network import TwoLayerNet

def load_mnist_data():
    """加载MNIST数据集"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cleaned data')
    
    # 加载数据
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # 归一化
    # x_train = x_train.astype(np.float32) / 255.0
    # x_test = x_test.astype(np.float32) / 255.0
    
    # 确保标签是整数类型
    # y_train = y_train.astype(np.int32)
    # y_test = y_test.astype(np.int32)
    
    # # 转换为one-hot编码
    # def _to_one_hot(y, num_classes=10):
    #     y = y.astype(np.int32)
    #     n_samples = len(y)
    #     one_hot = np.zeros((n_samples, num_classes))
    #     for i in range(n_samples):
    #         one_hot[i, y[i]] = 1
    #     return one_hot
    
    # y_train = _to_one_hot(y_train)
    # y_test = _to_one_hot(y_test)
    
    return (x_train, y_train), (x_test, y_test)

def predict_digit(network, image):
    """预测单个图像的数字"""
    x = image.reshape(1, 784)
    y = network.predict(x)
    predicted_digit = np.argmax(y)
    confidence = y[0][predicted_digit]
    return predicted_digit, confidence

def visualize_prediction(image, predicted_digit, confidence):
    """可视化图像和预测结果"""
    img_display = image.reshape(28, 28)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display, cmap='gray')
    plt.title(f'Predicted: {predicted_digit}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    
def main():
    # 加载数据
    (x_train, t_train), (x_test, t_test) = load_mnist_data()
    print("数据加载完成！")
    print(f"x_train shape: {x_train.shape}")
    print(f"t_train shape: {t_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"t_test shape: {t_test.shape}")  
    # 初始化网络
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    print("开始训练...")
    # 正确使用train方法
    network.train(x_train, t_train, 
                 learning_rate=0.06,
                 batch_size=100,
                 epochs=20)
    
    print("训练完成！")
    
    # 随机选择一个测试图像
    test_index = np.random.randint(0, len(x_test))
    test_image = x_test[test_index]
    true_label = np.argmax(t_test[test_index])
    
    # 预测和显示
    predicted_digit, confidence = predict_digit(network, test_image)
    print(f"\nTrue digit: {true_label}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2%}")
    visualize_prediction(test_image, predicted_digit, confidence)

if __name__ == '__main__':
    main()