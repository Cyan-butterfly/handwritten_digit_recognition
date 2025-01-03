import os
import numpy as np
from scripts.data_preprocessing import load_mnist_with_tensorflow, preprocess_data
from model.network import TwoLayerNet
import yaml
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_config():
    """加载配置文件"""
    config_path = os.path.join('configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def train_model(model, x_train, y_train, x_test, y_test, config):
    """训练模型"""
    train_size = x_train.shape[0]
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['epochs']
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(epochs):
        # 打乱数据
        idx = np.random.permutation(train_size)
        x_train = x_train[idx]
        y_train = y_train[idx]
        
        total_loss = 0
        for i in range(0, train_size, batch_size):
            batch_x = x_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            
            # 计算梯度并更新参数
            grads = model.gradient(batch_x, batch_y)
            for key in ('W1', 'b1', 'W2', 'b2'):
                model.params[key] -= learning_rate * grads[key]
                
            loss = model.loss(batch_x, batch_y)
            total_loss += loss
        
        # 计算并记录训练指标
        train_loss = total_loss / (train_size / batch_size)
        train_acc = model.accuracy(x_train, y_train)
        test_acc = model.accuracy(x_test, y_test)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        logging.info(f'Epoch {epoch + 1} / {epochs}')
        logging.info(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}')

    return train_loss_list, train_acc_list, test_acc_list

def main():
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 加载配置
    config = load_config()
    logging.info("Configuration loaded")
    
    # 加载并预处理数据
    logging.info("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_with_tensorflow()
    x_train, x_test, y_train, y_test = preprocess_data(x_train, x_test, y_train, y_test)
    logging.info("Data preprocessing completed")
    
    # 创建模型
    model = TwoLayerNet(
        input_size=784,  # 28x28
        hidden_size=config['model']['hidden_size'],
        output_size=10,  # 10个数字类别
        weight_init_std=config['model']['weight_init_std']
    )
    logging.info("Model initialized")
    
    # 训练模型
    logging.info("Starting training...")
    start_time = time.time()
    train_loss_list, train_acc_list, test_acc_list = train_model(
        model, x_train, y_train, x_test, y_test, config
    )
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # 最终评估
    final_train_acc = model.accuracy(x_train, y_train)
    final_test_acc = model.accuracy(x_test, y_test)
    logging.info(f"Final training accuracy: {final_train_acc:.4f}")
    logging.info(f"Final test accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main()