import os
import struct
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import yaml

# 从 IDX 文件加载数据的函数
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# 通过 TensorFlow 加载数据的函数
def load_mnist_with_tensorflow():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

# 数据预处理函数
def preprocess_data(x_train, x_test, y_train, y_test):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test

# 保存数据为 .npy 文件的函数
def save_data_to_npy(data_dir, x_train, y_train, x_test, y_test):
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    print(f"数据已保存至 {data_dir}")

# 加载配置文件的函数
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 自动选择加载方法的主函数
def load_and_save_mnist_data(config_path):
    # 加载配置
    config = load_config(config_path)

    # 获取路径配置
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']

    # 检查 .npy 文件是否已经存在
    npy_files = ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']
    if all(os.path.exists(os.path.join(processed_data_path, file)) for file in npy_files):
        print("检测到已存在的 .npy 文件，跳过加载和保存步骤。")
        return

    # 自动选择加载方法
    train_images_path = os.path.join(raw_data_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(raw_data_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(raw_data_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(raw_data_path, "t10k-labels.idx1-ubyte")

    paths_to_check = [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path
    ]
    
    if all(os.path.exists(path) for path in paths_to_check):
        # 如果 IDX 文件存在，使用 struct 方法加载数据
        print("检测到 IDX 文件，使用 struct 方法加载数据...")
        x_train = load_mnist_images(train_images_path)
        y_train = load_mnist_labels(train_labels_path)
        x_test = load_mnist_images(test_images_path)
        y_test = load_mnist_labels(test_labels_path)
    else:
        # 否则，回退到 TensorFlow 方法加载数据
        print("未检测到 IDX 文件，使用 TensorFlow 方法加载数据...")
        x_train, y_train, x_test, y_test = load_mnist_with_tensorflow()

    # 数据预处理
    x_train, x_test, y_train, y_test = preprocess_data(x_train, x_test, y_train, y_test)

    # 保存为 .npy 格式
    save_data_to_npy(processed_data_path, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    # 配置文件路径
    config_path = r"E:\BaiduSyncdisk\mywork\aiworks\dl_exp\handwritten_digit_recognition\configs\config.yaml"

    # 自动选择加载方式并处理数据
    load_and_save_mnist_data(config_path=config_path)
