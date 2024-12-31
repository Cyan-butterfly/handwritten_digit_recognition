#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
更丰富的项目目录结构生成脚本

功能概述:
1. 递归遍历指定的项目目录, 自动生成 Markdown 文件以展示目录结构。
2. 使用丰富的 Emoji 图标对不同类型的文件进行区分, 视觉更直观。
3. 允许通过 exclude_dirs / exclude_files 排除不想展示的目录或文件。
4. 防止重复遍历 (如符号链接)。
5. 可与 pre-commit 钩子或 CI/CD 流程结合, 实现自动更新目录结构。

使用方法:
1. 在项目根目录运行:
      python generate_directory_structure.py
2. 查看生成的 DIRECTORY_STRUCTURE.md
3. 根据需要修改 ICONS 字典 或 if-elif 条件, 自定义更多图标 & 文件类型。

提示:
- 在不同的操作系统、终端或字体下, 某些 Emoji 可能无法正常显示或显示为方块。
  可根据需要替换为兼容性更好的字符, 或简单的 [文件夹]/[文件] 标记。
"""

import os
import subprocess
from pathlib import Path

# ===================== 图标映射 =====================
# 可根据需要进行增删改查，满足更多文件类型区分。
ICONS = {
    "folder": "📁",       # 文件夹
    "file": "📄",         # 通用文件
    "python": "🐍",       # Python脚本
    "docker": "🐳",       # Docker相关文件
    "image": "🖼️",        # 图片文件
    "video": "🎥",        # 视频文件
    "audio": "🔊",        # 音频文件
    "notebook": "📓",     # Jupyter Notebook
    "script": "📜",       # 其他脚本 (sh/bat等)
    "config": "⚙️",       # 配置文件 (.yaml/.yml/.ini/.conf/.json等)
    "doc": "📝",          # 文档文件 (.doc/.docx/.txt/.md等)
    "ppt": "📽",          # PPT 幻灯片 (.ppt/.pptx等)
    "pdf": "📕",          # PDF 文档
    "excel": "📊",        # Excel表格或数据文件 (.xls/.xlsx/.csv等)
    "exe": "💾",          # 可执行文件 (如.exe)
    "archive": "📦",      # 压缩包/归档文件 (.zip/.tar/.rar等)
    "other": "❓",        # 其他未知类型
}


def get_icon(file_path: Path) -> str:
    """
    根据文件类型(后缀/文件名)返回相应的图标。

    - 使用 suffix.lower() 识别后缀名, 对常见格式分配对应Emoji。
    - 使用 name.lower() 针对无后缀文件(如 Dockerfile, Makefile, README等)进行特殊处理。
    - 如果未匹配, 返回 ICONS["other"] 默认值。

    提示:
    - 若想进一步细分 json/yaml/ini 等文件, 可在这里独立判断并分配独立图标。
    - 不同系统对Emoji的渲染可能不一致, 可替换为更通用字符。
    """
    if file_path.is_dir():
        return ICONS["folder"]

    suffix = file_path.suffix.lower()
    name = file_path.name.lower()

    # ======= 常见脚本与可执行文件 =======
    if suffix == ".py":
        return ICONS["python"]
    elif suffix in [".sh", ".bat"]:
        return ICONS["script"]
    elif suffix == ".exe":
        return ICONS["exe"]

    # ======= 文档相关 =======
    elif suffix in [".md", ".txt", ".rst", ".log"]:
        return ICONS["doc"]
    elif suffix in [".doc", ".docx"]:
        return ICONS["doc"]
    elif suffix == ".pdf":
        return ICONS["pdf"]
    elif suffix in [".ppt", ".pptx", ".key"]:
        return ICONS["ppt"]
    elif suffix in [".xls", ".xlsx", ".csv"]:
        return ICONS["excel"]

    # ======= 图片/视频/音频 =======
    elif suffix in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"]:
        return ICONS["image"]
    elif suffix in [".mp4", ".avi", ".mov", ".mkv", ".flv"]:
        return ICONS["video"]
    elif suffix in [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"]:
        return ICONS["audio"]

    # ======= Notebook =======
    elif suffix == ".ipynb":
        return ICONS["notebook"]

    # ======= 配置 / 数据 / 压缩包 =======
    elif name in ["dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
        return ICONS["docker"]
    elif suffix in [".yaml", ".yml", ".json", ".ini", ".conf"]:
        return ICONS["config"]
    elif suffix in [".zip", ".tar", ".gz", ".rar", ".7z"]:
        return ICONS["archive"]

    # ======= 其他文件名特殊处理: Makefile, LICENSE, README等 =======
    elif name == "makefile":
        # 如果您想给 Makefile 特殊图标, 可以改成 ICONS["script"] 或自定义
        return ICONS["script"]
    elif name in ["license", "readme"]:
        return ICONS["doc"]

    # ======= 未匹配到的情况 =======
    return ICONS["other"]


def generate_markdown(
    dir_path: Path,
    indent: int = 0,
    exclude_dirs=None,
    exclude_files=None,
    visited=None,
) -> str:
    """
    递归生成带有图标和格式化的 Markdown 目录结构。

    参数:
    - dir_path (Path): 要遍历的目录路径
    - indent (int): 缩进层级, 用于生成嵌套列表
    - exclude_dirs (list): 要排除的目录列表(精确匹配目录名)
    - exclude_files (list): 要排除的文件列表(精确匹配文件名)
    - visited (set): 已访问过的目录集合, 防止重复遍历(例如循环引用)

    返回值:
    - markdown (str): 拼接生成的目录结构字符串, 用于写入 Markdown 文件

    提示:
    - 您可以根据需要继续改进 exclude_dirs / exclude_files,
      如使用通配符或正则表达式来排除特定模式。
    """

    if exclude_dirs is None:
        exclude_dirs = []
    if exclude_files is None:
        exclude_files = []
    if visited is None:
        visited = set()

    markdown = ""
    resolved_dir = dir_path.resolve()

    # 如果已访问过该目录, 直接返回空字符串, 避免重复遍历
    if resolved_dir in visited:
        return markdown
    visited.add(resolved_dir)

    # 按 "文件夹优先, 名称排序" 排列
    items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

    for item in items:
        # 排除指定的目录或文件 (精确匹配)
        if item.is_dir() and item.name in exclude_dirs:
            continue
        if item.is_file() and item.name in exclude_files:
            continue

        icon = get_icon(item)

        # 如果是文件夹, 在名称后加 "/" 以示区分
        if item.is_dir():
            line = "  " * indent + f"- {icon} **{item.name}**/"
            markdown += line + "\n"
            # 递归遍历子目录
            markdown += generate_markdown(
                dir_path=item,
                indent=indent + 1,
                exclude_dirs=exclude_dirs,
                exclude_files=exclude_files,
                visited=visited,
            )
        else:
            # 如果是文件
            line = "  " * indent + f"- {icon} **{item.name}**"
            markdown += line + "\n"

    return markdown


def main():
    """
    主函数:
    1. 获取当前工作目录作为项目根目录
    2. 设置要排除的目录和文件 (如 .git、__pycache__、.idea 等)
    3. 生成目录结构字符串
    4. 写入到 DIRECTORY_STRUCTURE.md
    5. 可与 pre-commit 等集成, 在每次提交前自动刷新
    """
    base_dir = Path.cwd()

    # 根据需要排除一些不需要展示的目录 / 文件
    exclude_dirs = [".git", "__pycache__", ".idea", ".vscode", "node_modules"]
    exclude_files = ["generate_directory_structure.py", "Pipfile", "Pipfile.lock"]

    # 生成目录结构
    directory_structure = generate_markdown(
        dir_path=base_dir,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
    )

    # 写入 Markdown 文件
    output_file = base_dir / "DIRECTORY_STRUCTURE.md"
    with output_file.open("w", encoding="utf-8") as f:
        f.write("# 项目目录结构\n\n")
        f.write(directory_structure)

    print(f"目录结构已生成并写入到 {output_file}\n")


if __name__ == "__main__":
    main()
