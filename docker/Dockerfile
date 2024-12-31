
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
