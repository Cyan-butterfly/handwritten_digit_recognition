import numpy as np
import matplotlib.pyplot as plt
from network import Network
import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import sys
from datetime import datetime
import time
from scipy import stats
import pandas as pd
from network import Network

def load_mnist():
    """
    加载MNIST数据集
    返回: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    try:
        # 尝试从本地加载数据
        data_dir = "data/mnist"
        
        print("Loading MNIST dataset...")
        
        # 加载训练数据
        train_images = np.load(os.path.join(data_dir, 'train_images.npy'))
        train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
        
        # 加载测试数据
        test_images = np.load(os.path.join(data_dir, 'test_images.npy'))
        test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
        
        # 数据预处理
        # 1. 将像素值归一化到 0-1 之间
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        
        # 2. 将图像展平为一维数组
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
        
        print(f"Dataset loaded successfully:")
        print(f"Training set: {train_images.shape[0]} images")
        print(f"Test set: {test_images.shape[0]} images")
        
        return train_images, train_labels, test_images, test_labels
        
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        print("Using random data for testing...")
        
        # 如果加载失败，生成随机数据用于测试
        train_images = np.random.randn(60000, 784) * 0.1
        train_labels = np.random.randint(0, 10, 60000)
        test_images = np.random.randn(10000, 784) * 0.1
        test_labels = np.random.randint(0, 10, 10000)
        
        return train_images, train_labels, test_images, test_labels
def run_experiment(hidden_size, learning_rate, X_train, y_train, X_test, y_test, run_num, logger=None):
    """运行单次实验"""
    # 创建实验目录
    exp_name = f"exp_h{hidden_size}_lr{learning_rate}_run{run_num}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 初始化日志记录器
    logger = ExperimentLogger(exp_dir)
    logger.log_experiment_start({
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "run_num": run_num
    })
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化网络
    net = Network(hidden_size=hidden_size, learning_rate=learning_rate)
    
    # 训练网络, 传入logger
    history = net.train(X_train, y_train, epochs=10, batch_size=32, logger=logger)
    
    # 记录训练时间
    training_time = time.time() - start_time
    
    # 保存模型
    save_model(net, exp_dir)
    
    # 记录最终结果
    final_metrics = {
        "final_accuracy": history['accuracy'][-1],
        "final_loss": history['loss'][-1],
        "training_time": training_time
    }
    logger.log_experiment_end(final_metrics)
    
    # 保存训练历史
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(history, exp_dir)
    
    return history

def plot_training_curves(history, exp_dir):
    """为单个实验绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"))
    plt.close()


class ExperimentLogger:
    """实验日志记录器"""
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.log_path = os.path.join(exp_dir, "training.log")
        
        # 确保实验目录存在
        os.makedirs(exp_dir, exist_ok=True)
        
        # 配置日志记录器
        self.logger = logging.getLogger(f"experiment_{exp_dir}")
        self.logger.setLevel(logging.INFO)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_experiment_start(self, config):
        """记录实验开始信息"""
        self.logger.info("="*50)
        self.logger.info("实验开始")
        self.logger.info(f"配置信息: {json.dumps(config, indent=2)}")
        
    def log_epoch(self, epoch, metrics):
        """记录每个epoch的信息"""
        self.logger.info(
            f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )
    
    def log_experiment_end(self, final_metrics):
        """记录实验结束信息"""
        self.logger.info("实验结束")
        self.logger.info(f"最终指标: {json.dumps(final_metrics, indent=2)}")
        self.logger.info("="*50)

class ExperimentAnalyzer:
    """实验结果分析器"""
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def calculate_statistics(self, histories):
        """计算实验统计信息"""
        accuracies = [h['accuracy'][-1] for h in histories]  # 使用最后一个epoch的准确率
        losses = [h['loss'][-1] for h in histories]
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        # 计算95%置信区间
        confidence = 0.95
        degrees_of_freedom = len(accuracies) - 1
        
        if degrees_of_freedom > 0:
            accuracy_ci = stats.t.interval(
                confidence,
                degrees_of_freedom,
                loc=mean_accuracy,
                scale=std_accuracy/np.sqrt(len(accuracies))
            )
            loss_ci = stats.t.interval(
                confidence,
                degrees_of_freedom,
                loc=mean_loss,
                scale=std_loss/np.sqrt(len(losses))
            )
        else:
            # 如果只有一次运行，使用点估计
            accuracy_ci = (mean_accuracy, mean_accuracy)
            loss_ci = (mean_loss, mean_loss)
        
        return {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "accuracy_ci": accuracy_ci,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "loss_ci": loss_ci
        }
    
    def _calculate_convergence_epochs(self, config_results):
        """计算收敛所需的epoch数"""
        convergence_epochs = []
        for result in config_results:
            final_acc = result['accuracy'][-1]
            threshold = 0.9 * final_acc
            epochs = np.where(np.array(result['accuracy']) >= threshold)[0]
            convergence_epochs.append(epochs[0] if len(epochs) > 0 else len(result['accuracy']))
        return np.mean(convergence_epochs)

def save_model(model, exp_dir):
    """保存模型权重"""
    model_dir = os.path.join(exp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存权重
    weights_path = os.path.join(model_dir, "weights.npz")
    np.savez(weights_path,
             W1=model.W1,
             b1=model.b1,
             W2=model.W2,
             b2=model.b2)
    
    # 保存模型配置
    config_path = os.path.join(model_dir, "config.json")
    config = {
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "output_size": model.output_size,
        "learning_rate": model.learning_rate
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_model(model_dir):
    """加载模型权重"""
    # 加载配置
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model = Network(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        learning_rate=config["learning_rate"]
    )
    
    # 加载权重
    weights = np.load(os.path.join(model_dir, "weights.npz"))
    model.W1 = weights['W1']
    model.b1 = weights['b1']
    model.W2 = weights['W2']
    model.b2 = weights['b2']
    
    return model



def main():
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 创建实验主目录
    os.makedirs("experiments", exist_ok=True)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist()
    
    # 实验配置
    hidden_sizes = [25, 50, 100]
    learning_rates = [0.01, 0.1, 0.5]
    runs_per_config = 3
    # 计算实验总数
    total_experiments = len(hidden_sizes) * len(learning_rates) * runs_per_config
    current_experiment = 0  # 初始化计数器
   # 初始化分析器
    analyzer = ExperimentAnalyzer("experiments")
    
    # 存储所有实验结果
    results = {}
    detailed_stats = {}
    
    # 运行实验
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            config_results = []
            for run in range(runs_per_config):
                print(f"\nExperiment {current_experiment}/{total_experiments}")
                print(f"Configuration: hidden_size={hidden_size}, lr={lr}, run={run+1}")
                
                # 创建logger实例
                exp_dir = os.path.join("experiments", f"exp_h{hidden_size}_lr{lr}_run{run+1}")
                logger = ExperimentLogger(exp_dir)
                
                # 运行实验，传入logger
                history = run_experiment(
                    hidden_size, lr, X_train, y_train, X_test, y_test, run+1, logger
                )
                config_results.append(history)
                
            # 添加回统计计算代码
            config_key = f"h{hidden_size}_lr{lr}"
            detailed_stats[config_key] = analyzer.calculate_statistics(config_results)
            results[config_key] = detailed_stats[config_key]
    
    # 保存详细统计结果
    with open("experiments/detailed_stats.json", "w") as f:
        json.dump(detailed_stats, f, indent=4)
    
    # 创建比较图
    create_comparison_plots(results, hidden_sizes, learning_rates)
    
    print("\nExperiments completed. Results saved in experiments/detailed_stats.json")
    # 生成实验报告
    generate_experiment_report(results, hidden_sizes, learning_rates, start_time)

    # 保存实验配置
    experiment_config = {
        "hidden_sizes": hidden_sizes,
        "learning_rates": learning_rates,
        "runs_per_config": runs_per_config,
        "epochs": 10,
        "batch_size": 32
    }
    with open("experiments/experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)

    total_experiments = len(hidden_sizes) * len(learning_rates) * runs_per_config
    current_experiment = 0
    
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for run in range(runs_per_config):
                current_experiment += 1
                print(f"\nExperiment {current_experiment}/{total_experiments}")
                print(f"Configuration: hidden_size={hidden_size}, lr={lr}, run={run+1}")

def create_comparison_plots(results, hidden_sizes, learning_rates):
    """
    创建实验结果的对比可视化
    """
    # 设置图表样式
    plt.style.use('seaborn')
    
    # 创建一个2x2的子图布局
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 不同隐藏层大小的性能对比
    ax1 = plt.subplot(2, 2, 1)
    for hidden_size in hidden_sizes:
        accuracies = [results[f"h{hidden_size}_lr{lr}"]["avg_accuracy"] for lr in learning_rates]
        ax1.plot(learning_rates, accuracies, 'o-', label=f'Hidden Size={hidden_size}')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy vs Learning Rate')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 不同学习率的性能对比
    ax2 = plt.subplot(2, 2, 2)
    for lr in learning_rates:
        accuracies = [results[f"h{h}_lr{lr}"]["avg_accuracy"] for h in hidden_sizes]
        ax2.plot(hidden_sizes, accuracies, 'o-', label=f'Learning Rate={lr}')
    ax2.set_xlabel('Hidden Layer Size')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Accuracy vs Hidden Layer Size')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 热力图显示准确率
    ax3 = plt.subplot(2, 2, 3)
    accuracy_matrix = np.zeros((len(hidden_sizes), len(learning_rates)))
    for i, h in enumerate(hidden_sizes):
        for j, lr in enumerate(learning_rates):
            accuracy_matrix[i, j] = results[f"h{h}_lr{lr}"]["avg_accuracy"]
    im = ax3.imshow(accuracy_matrix, cmap='YlOrRd')
    plt.colorbar(im)
    ax3.set_xticks(range(len(learning_rates)))
    ax3.set_yticks(range(len(hidden_sizes)))
    ax3.set_xticklabels(learning_rates)
    ax3.set_yticklabels(hidden_sizes)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Hidden Layer Size')
    ax3.set_title('Accuracy Heat Map')
    
    # 4. 损失值对比
    ax4 = plt.subplot(2, 2, 4)
    for hidden_size in hidden_sizes:
        losses = [results[f"h{hidden_size}_lr{lr}"]["avg_loss"] for lr in learning_rates]
        ax4.plot(learning_rates, losses, 'o-', label=f'Hidden Size={hidden_size}')
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Average Loss')
    ax4.set_title('Loss vs Learning Rate')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments/comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_experiment_report(results, hidden_sizes, learning_rates, start_time):
    """
    生成实验报告
    """
    # 找出最佳配置
    best_config = max(results.items(), key=lambda x: x[1]['avg_accuracy'])
    best_config_name = best_config[0]
    best_config_metrics = best_config[1]
    
    # 创建报告目录
    report_dir = "experiment_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"experiment_report_{timestamp}.md")
    
    # 计算实验时长
    duration = datetime.now() - start_time
    
    with open(report_path, "w", encoding='utf-8') as f:
        # 写入报告头部
        f.write("# 手写数字识别实验报告\n\n")
        
        # 实验概述
        f.write("## 实验概述\n")
        f.write(f"- 实验日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 实验时长：{duration}\n")
        f.write("- 实验目的：探究不同网络配置对手写数字识别性能的影响\n")
        f.write("\n### 实验参数配置\n")
        f.write(f"- 隐藏层大小：{hidden_sizes}\n")
        f.write(f"- 学习率：{learning_rates}\n")
        f.write("- 每组配置重复次数：3\n\n")
        
        # 最佳配置结果
        f.write("## 最佳配置结果\n")
        hidden_size = int(best_config_name.split('_')[0][1:])
        learning_rate = float(best_config_name.split('_')[1][2:])
        f.write(f"- 最佳隐藏层大小：{hidden_size}\n")
        f.write(f"- 最佳学习率：{learning_rate}\n")
        f.write(f"- 最高准确率：{best_config_metrics['avg_accuracy']:.4f}\n")
        f.write(f"- 最低损失值：{best_config_metrics['avg_loss']:.4f}\n\n")
        
        # 详细实验数据
        f.write("## 详细实验数据\n\n")
        f.write("### 所有配置结果\n")
        f.write("| 隐藏层大小 | 学习率 | 平均准确率 | 平均损失 |\n")
        f.write("|------------|---------|------------|----------|\n")
        
        for h in hidden_sizes:
            for lr in learning_rates:
                config_key = f"h{h}_lr{lr}"
                metrics = results[config_key]
                f.write(f"| {h} | {lr} | {metrics['avg_accuracy']:.4f} | {metrics['avg_loss']:.4f} |\n")
        
        # 可视化结果
        f.write("\n## 可视化结果\n")
        f.write("### 性能对比图\n")
        f.write("![实验结果对比图](../experiments/comparison_plots.png)\n\n")
        
        # 结论与建议
        f.write("## 结论与建议\n\n")
        f.write("### 主要发现\n")
        f.write(f"1. 最佳性能配置为隐藏层大小 {hidden_size}，学习率 {learning_rate}\n")
        f.write(f"2. 平均准确率达到 {best_config_metrics['avg_accuracy']:.4f}\n")
        
        # 附录
        f.write("\n## 附录\n\n")
        f.write("### 实验环境\n")
        f.write(f"- Python版本：{sys.version.split()[0]}\n")
        f.write("- 主要依赖包版本：\n")
        f.write(f"  - NumPy: {np.__version__}\n")
        f.write(f"  - Matplotlib: {plt.__version__}\n")
        
        print(f"\nExperiment report generated: {report_path}")


if __name__ == "__main__":
    main()