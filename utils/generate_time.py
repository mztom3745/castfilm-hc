from datetime import datetime

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def get_current_time():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间为 *年*月*日*时*分*秒
    formatted_time = now.strftime("%Y年%m月%d日%H时%M分%S秒")
    return formatted_time

def get_current_time2():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间为 YYYY_M_D_H_M_S
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    return formatted_time



# 增加毫秒, 保留四位,
def get_current_time3():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间为 YYYY_M_D_H_M_S_MS，保留四位毫秒
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:]  # 取前四位毫秒
    return formatted_time

def array_to_qpixmap(array):
    if isinstance(array, Image.Image):
        array = np.array(array)
    array = array.astype(np.uint8)
    if array.ndim == 2:
        # 在第三个维度上添加一个维度
        array = np.expand_dims(array, axis=2)
    height, width, channels = array.shape
    if channels == 1:
        format = QImage.Format_Grayscale8
    elif channels == 3:
        # RGB模式
        format = QImage.Format_RGB888
    elif channels == 4:
        # RGBA模式
        format = QImage.Format_RGBA8888
    else:
        raise ValueError("Unsupported array shape")
    qimage = QImage(array.data, width, height, width * channels, format)
    # cv2.imwrite('C:\\Users\\tf\\Desktop\\output_with_rectangles-array.jpg', QImage)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap
