import cv2
import numpy as np
import os
from utils.log_util import logger
from detect_core.defect_config import DefectConfig
from config.constant import Light

def safe_imread(path, flags=cv2.IMREAD_GRAYSCALE):
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, flags)


def light_judge(
    input_image,
    background_range=(200, 225),
    num_slices=25,
    defect_gray_threshold=190,
    left=DefectConfig.LEFT_EDGE_X,
    right=DefectConfig.RIGHT_EDGE_X,
    too_dark=0.2,  # 百分比阈值（%）
    too_light=50,  # 百分比阈值（%）
    num_chunk=5    # 超过多少块时报警
):
    """
    光照稳定性检测，输出综合判断结果
    """
    # 读图
    if isinstance(input_image, str):  # 传入路径
        img = safe_imread(input_image, cv2.IMREAD_GRAYSCALE)
    else:  # 传入已读灰度图
        img = input_image

    if img is None:
        raise ValueError(f"无法读取图像: {input_image}")

    height, width = img.shape
    left, right = max(0, left), min(width, right)
    region = img[:, left:right]
    region_width = right - left
    slice_width = max(region_width // num_slices, 1)

    too_dark_count = 0
    too_light_count = 0

    for i in range(num_slices):
        x_start = left + i * slice_width
        x_end = min(left + (i + 1) * slice_width, right)
        slice_img = region[:, x_start - left:x_end - left]
        if slice_img.size == 0:
            continue

        # 计算背景占比 & 缺陷占比
        bg_mask = (slice_img >= background_range[0]) & (slice_img <= background_range[1])
        bg_ratio = np.sum(bg_mask) / slice_img.size * 100.0
        defect_ratio = np.sum(slice_img < defect_gray_threshold) / slice_img.size * 100.0

        if defect_ratio > too_dark:
            too_dark_count += 1
        if bg_ratio < too_light:
            too_light_count += 1

    # 生成判断结果
    messages = []
    if too_dark_count > num_chunk:
        messages.append("背景颜色过深，无效缺陷区域划分过多，建议调正光照强度后重新检测")
    if too_light_count > num_chunk:
        messages.append("背景颜色过浅，亮度过高的区域过多，建议调正光照强度后重新检测")

    if not messages:
        return Light.NORMAL

    return "；".join(messages)


# 示例用法
if __name__ == "__main__":
    img_path = r"D:\guanwei\castfilm4090\cast-film\test_data\16384_4015\20250829_140433_4b1b9dcf.jpg"
    result = light_judge(
        img_path,
        background_range=(200, 225),
        num_slices=25,
        defect_gray_threshold=190,
        too_dark=0.2,
        too_light=50,
        num_chunk=5
    )
    print("检测结果:", result)
    if result != Light.NORMAL:
        logger.warning(f"检测结果:{result}")
    else:
        logger.info(f"检测结果:{result}")
