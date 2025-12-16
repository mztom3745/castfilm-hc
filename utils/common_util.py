import asyncio
import base64
import io
import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image



def modify_ccf_file(input_file_path, new_value, output_dir=None):
    """
    修改CCF文件中的Crop Height和Scale Vertical值

    Args:
        input_file_path (str): 输入CCF文件路径
        new_value (int): 新的高度值
        output_dir (str): 输出目录，默认为输入文件同目录

    Returns:
        str: 新文件路径
    """
    # 读取原文件内容
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用字符串替换方式替代正则表达式，避免转义问题
    # 替换Crop Height的值
    content = re.sub(r'Crop Height=\d+', f'Crop Height={new_value}', content)

    # 替换Scale Vertical的值
    content = re.sub(r'Scale Vertical=\d+', f'Scale Vertical={new_value}', content)

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_file_path)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 创建新的文件名，以值命名
    new_file_name = f"{new_value}.ccf"
    new_file_path = os.path.join(output_dir, new_file_name)

    # 写入修改后的内容到新文件
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"已创建新CCF文件: {new_file_path}")
    print(f"Crop Height 和 Scale Vertical 已修改为: {new_value}")

def write_config_txt(ccf, pixel, output_dir):
    """
    将两个值写入txt文件，文件名按照指定格式命名

    Args:
        ccf: 第一个值
        pixel: 第二个值（浮点数）
        output_dir (str): 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理pixel值，去除小数点，保留所有数字
    pixel_formatted = f"{pixel:.4f}"
    pixel_filename = pixel_formatted.replace('.', '')

    # 创建文件名
    filename = f"ccf_{ccf}_pixel_{pixel_filename}.txt"
    file_path = os.path.join(output_dir, filename)

    # 将值写入文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"Value ccf: {ccf}\n")
        file.write(f"Value pixel: {pixel_formatted}\n")

    print(f"已创建文件: {file_path}")
    print(f"写入的值: {ccf}, {pixel_formatted}")

def read_latest_pixel(path):
    """
    读取目录下最新的txt文件，提取Value pixel:后的值

    Args:
        path (str): 要搜索的目录路径

    Returns:
        float: 提取到的pixel值，如果未找到则返回None
    """
    # 获取目录下所有txt文件
    txt_files = list(Path(path).glob("*.txt"))

    if not txt_files:
        return None

    # 找到最新的文件（按修改时间）
    latest_file = max(txt_files, key=lambda f: f.stat().st_mtime)

    # 读取文件内容
    with open(latest_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Value pixel:"):
                # 提取冒号后的值并转换为浮点数
                pixel_value_str = line.split(":")[1].strip()
                try:
                    return float(pixel_value_str)
                except ValueError:
                    return None

    return None

# ------------------- 异步保存工具函数 -------------------

def split_image_to_base64(pil_image, split_axis=0, quality=100):
        image = np.array(pil_image)
        height = image.shape[0]
        new_width = int(16384 / 10)
        new_height = int(height / 10)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # 如果输入是 numpy 数组，先转为 PIL.Image
        if isinstance(image, np.ndarray):
            # 确保numpy数组的数据类型正确
            if image.dtype != np.uint8:
                # 归一化到0-255范围
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("输入必须是 PIL.Image 或 numpy 数组")

        width, height = image.size

        if split_axis == 0:
            # 垂直均分（横向切分），确保覆盖整个宽度
            part_width = width // 3
            parts = [
                image.crop((0, 0, part_width, height)),
                image.crop((part_width, 0, 2 * part_width, height)),
                image.crop((2 * part_width, 0, width, height))
            ]
        else:
            # 水平均分（纵向切分）
            part_height = height // 3
            parts = [
                image.crop((0, 0, width, part_height)),
                image.crop((0, part_height, width, 2 * part_height)),
                image.crop((0, 2 * part_height, width, height))
            ]

        # 将每一部分转为 Base64
        base64_parts = []
        for part in parts:
            buffered = io.BytesIO()
            # 使用更高的质量减少压缩伪影
            part.save(buffered, format="JPEG", quality=quality, optimize=True)

            img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_parts.append(img_base64)
        return base64_parts

