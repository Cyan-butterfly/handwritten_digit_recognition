import os
from pathlib import Path
import subprocess

def create_directory(path):
    """
    创建目录，如果目录不存在则创建。
    
    参数:
    - path (Path): 要创建的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {path}")

def create_file(path, content=""):
    """
    创建文件并写入内容，如果文件已存在则跳过。
    
    参数:
    - path (Path): 要创建的文件路径
    - content (str): 要写入文件的内容
    """
    if not path.exists():
        with path.open('w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建文件: {path}")
    else:
        print(f"文件已存在: {path}")

def create_project_structure(base_dir, structure):
    """
    递归创建项目目录结构和文件。
    
    参数:
    - base_dir (Path): 项目根目录路径
    - structure (dict): 项目结构字典
    """
    for name, content in structure.items():
        path = base_dir / name
        if isinstance(content, dict):
            # 如果内容是字典，说明是文件夹
            create_directory(path)
            # 递归调用
            create_project_structure(path, content)
        else:
            # 如果内容不是字典，说明是文件
            create_file(path, content)

def initialize_git_repo(base_dir):
    """初始化Git仓库并添加初始提交。"""
    try:
        # 初始化Git仓库
        subprocess.run(['git', 'init'], cwd=base_dir, check=True)
        print("Git仓库已初始化。")
        
        # 创建 .gitignore 文件
        gitignore_path = base_dir / '.gitignore'
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
venv/
ENV/
env/

# Logs
logs/
*.log

# Data
data/raw data/
data/cleaned data/
data/*.csv

# Models
models/

# Misc
.DS_Store
*.swp
"""
        create_file(gitignore_path, gitignore_content)
        
        # 添加所有文件到Git
        subprocess.run(['git', 'add', '.'], cwd=base_dir, check=True)
        print("所有文件已添加到Git。")
        
        # 初始提交
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=base_dir, check=True)
        print("初始提交已完成。")
    except subprocess.CalledProcessError as e:
        print(f"Git操作失败: {e}")

def main():
    from pathlib import Path

    # 获取当前工作目录作为项目根目录
    base_directory = Path.cwd()
    
    # 创建项目目录结构和文件
    structure = {
        'data': {
            'raw data': {},
            'cleaned data': {},
            'data documentation': {
                'data documentation template.md': '# 数据说明文档模板\n\n描述数据集的结构、来源和预处理步骤。'
            },
            'data visualization': {
                'example chart.png': ''  # 可以手动添加图片
            }
        },
        'feature engineering': {
            'feature selection': {},
            'feature construction': {},
            'feature description document': {
                'feature description template.md': '# 特征描述文档模板\n\n描述所使用特征的生成方法和意义。'
            }
        },
        'model': {
            'training script': {
                'train_model.py': '# 训练模型的脚本\n\nimport torch\n\n# 模型训练代码'
            },
            'tuning results': {},
            'model saving': {},
            'model evaluation': {
                'evaluate_model.py': '# 模型评估脚本\n\n# 模型评估代码'
            },
            'pretrained models': {}  # 存放预训练模型
        },
        'evaluation report': {
            'metrics report': {
                'metrics report template.md': '# 指标报告模板\n\n记录模型的性能指标。'
            },
            'visualization charts': {
                'example chart.png': ''  # 可以手动添加图片
            },
            'comprehensive analysis report': {
                'comprehensive analysis report template.md': '# 综合分析报告模板\n\n对模型的整体表现进行分析。'
            }
        },
        'code': {
            'main code': {
                'main.py': '# 主程序\n\ndef main():\n    pass\n\nif __name__ == "__main__":\n    main()'
            },
            'module code': {
                '__init__.py': '',
                'data_loader.py': '# 数据加载模块\n\ndef load_data():\n    pass'
            },
            'utility scripts': {
                '__init__.py': '',
                'utils.py': '# 工具函数\n\ndef helper():\n    pass'
            },
            'configs': {
                'config.yaml': '# 配置文件模板\n\nlearning_rate: 0.001\nbatch_size: 32\nepochs: 100'
            }
        },
        'notebooks': {
            'exploratory_analysis.ipynb': ''  # 可以手动添加Jupyter Notebook
        },
        'scripts': {
            'data_preprocessing.py': '# 数据预处理脚本\n\ndef preprocess():\n    pass',
            'data_augmentation.py': '# 数据增强脚本\n\ndef augment_data():\n    pass'
        },
        'models': {},  # 存放训练好的模型
        'logs': {},  # 存放日志文件
        'tests': {
            'test_data_loader.py': '# 测试数据加载模块\n\ndef test_load_data():\n    pass'
        },
        'configs': {
            'config.yaml': '# 项目配置文件\n\nmodel:\n  type: "ResNet"\n  layers: 50\ntraining:\n  epochs: 100\n  batch_size: 32'
        },
        'docker': {
            'Dockerfile': '''
# 使用官方的Python基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器内
COPY . /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8080

# 运行主程序
CMD ["python", "main.py"]
''',
            'docker-compose.yml': '''
version: '3.8'

services:
  app:
    build: ./docker
    ports:
      - "8080:8080"
    volumes:
      - .:/app
    environment:
      - ENV=production
'''
        },
        'CI_CD': {
            '.github': {
                'workflows': {
                    'python-app.yml': '''
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest
'''
                }
            }
        },
        'documentation': {
            'project plan': {
                'project plan template.md': '# 项目计划模板\n\n描述项目的目标、方法和时间表。'
            },
            'meeting notes': {
                'meeting notes template.md': '# 会议记录模板\n\n记录会议讨论的内容和决策。'
            },
            'user manual': {
                'user manual template.md': '# 用户手册模板\n\n指导用户如何使用项目。'
            }
        },
        'sharing materials': {
            'PPT': {
                'example PPT.pptx': ''  # 可以手动添加PPT文件
            },
            'demo video': {
                'example video.mp4': ''  # 可以手动添加视频文件
            }
        },
        'experiment records': {
            'experiment 1': {
                'experiment 1 record.md': '# 实验1记录\n\n记录实验1的过程和结果。'
            },
            'experiment 2': {
                'experiment 2 record.md': '# 实验2记录\n\n记录实验2的过程和结果。'
            },
            'experiment summary': {
                'experiment summary template.md': '# 实验总结模板\n\n总结所有实验的发现和结论。'
            }
        },
        'environment': {
            'requirements.txt': '# 项目依赖\n\ntorch\nnumpy\npandas\nscikit-learn\n',
            'environment configuration instructions': {
                'configuration instructions.md': '# 环境配置说明\n\n说明如何设置项目的运行环境。'
            }
        },
        'README.md': '# 项目名称\n\n项目简介和说明。'
    }

    create_project_structure(base_directory, structure)
    
    # 初始化Git仓库并进行初始提交
    initialize_git_repo(base_directory)
    
    print("\n项目目录结构、示例文件和Git仓库已成功生成。")

if __name__ == "__main__":
    main()