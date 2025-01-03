import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from model.network import TwoLayerNet
from dataset.mnist import load_mnist

def predict_digit(network, image):
    """预测单个图像的数字
    
    Args:
        network: 训练好的网络
        image: 形状为(784,)的图像数据
        
    Returns:
        预测的数字和对应的概率
    """
    # 确保图像是正确的形状
    x = image.reshape(1, 784)
    
    # 获取预测结果
    y = network.predict(x)
    predicted_digit = np.argmax(y)
    confidence = y[0][predicted_digit]
    
    return predicted_digit, confidence

def visualize_prediction(image, predicted_digit, confidence):
    """可视化图像和预测结果
    
    Args:
        image: 形状为(784,)的图像数据
        predicted_digit: 预测的数字
        confidence: 预测的置信度
    """
    # 重塑图像为28x28
    img_display = image.reshape(28, 28)
    
    # 创建图像
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display, cmap='gray')
    plt.title(f'Predicted: {predicted_digit}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()

def main():
    """主函数"""
    # 加载MNIST数据集
    (_, _), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    
    # 加载训练好的网络
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    # 随机选择一个测试图像
    test_index = np.random.randint(0, len(x_test))
    test_image = x_test[test_index]
    true_label = np.argmax(t_test[test_index])
    
    # 预测
    predicted_digit, confidence = predict_digit(network, test_image)
    
    # 显示结果
    print(f"True digit: {true_label}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2%}")
    
    # 可视化
    visualize_prediction(test_image, predicted_digit, confidence)

if __name__ == '__main__':
    main()
