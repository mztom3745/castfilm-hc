import math
from config.constant import Machine_Learning, Path, Constants
import numpy as np

from utils.common_util import read_latest_pixel


class DefectConfig:
    # 像素与微米的换算率
    UM_PER_PIXEL = read_latest_pixel(Path.CCF_PIXEL_DIR) if read_latest_pixel(Path.CCF_PIXEL_DIR) is not None else Constants.UM_PER_PIXEL

    # 用于标记的缺陷直径边界（单位：um）
    # 对应 SIZE_LIST 的区间：[0,25), [25,50), [50,100), [100,200), [200,300), [300,400), [400,500), [500,600), [600,700), [700,800), [800,inf)
    DIAMETER_BINS_UM = [0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, float('inf')]

    # 预计算的像素面积阈值（根据直径计算的圆面积）
    PIXEL_AREA_BINS = []

    # 缺陷类别
    DEFECT_CLASSES = {
        '晶点': 0,
        '黑点': 0,
        '纤维': 0,
        '其它': 0
    }

    # 缺陷尺寸列表（与直径区间一一对应，便于统计显示）
    SIZE_LIST = ['_25', '25-50', '50-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700',
                 '700-800', 'O800']

    # 机器学习/分类相关参数（来源于全局配置）
    UPPER_BOUND = Machine_Learning.UPPER_BOUND
    DARK_BOUND = Machine_Learning.DARK_BOUND
    DARK_VALUE = Machine_Learning.DARK_VALUE
    # 黑点判断时使用的暗像素占比阈值，缺省更严格
    DARK_RATIO_THRESHOLD = getattr(Machine_Learning, "DARK_RATIO_THRESHOLD", 0.8)

    # 检测相关参数（统一去除魔法数字）
    DETECT_REF_GRAY = 210  # 参考灰度值
    # 动态参考灰度值范围
    REF_GRAY_MIN = 190
    REF_GRAY_MAX = 230
    DETECT_DIFF_THRESHOLD = 40  # 差异阈值
    DETECT_AREA_MIN = 8  # 缺陷最小面积（像素） 8像素一般对应直径23左右
    DETECT_AREA_MAX = 4000  # 缺陷最大面积（像素）
    EDGE_RATIO = 0.8  # 视为边缘线的宽/高占比阈值
    SMOOTH_PIXELS_THRESHOLD = 9  # 平滑像素阈值
    LEFT_EDGE_X = 2000
    RIGHT_EDGE_X = 11000

    # 高亮显示相关参数
    HIGHLIGHT_USE_FILL = True
    HIGHLIGHT_OUTLINE_COLOR = (255, 0, 0)
    HIGHLIGHT_FILL_COLOR = (255, 0, 0)

    @classmethod
    def calculate_dynamic_ref_gray(cls, first_image):
        """
        根据第一张图像计算动态参考灰度值
        :param first_image: 第一张图像
        :return: 动态参考灰度值
        """
        # 计算直方图
        hist = np.histogram(first_image, bins=256, range=(0, 256))[0]
        
        # 找到出现频率最高的像素值
        most_frequent_gray = np.argmax(hist)

        # 确保参考灰度值在合理范围内，不能偏离210太远
        clamped_gray = max(cls.REF_GRAY_MIN, min(cls.REF_GRAY_MAX, most_frequent_gray))
        
        return clamped_gray

    @classmethod
    def _calculate_pixel_areas(cls):
        """将直径换算为像素面积（以直径/2计算半径）"""
        size_bins = []
        for d in cls.DIAMETER_BINS_UM:
            if math.isinf(d):
                size_bins.append(float('inf'))
            else:
                r = d / 2
                pixel_area = math.pi * (r / cls.UM_PER_PIXEL) ** 2
                size_bins.append(round(pixel_area))
        return size_bins


# 在模块加载时预计算像素面积阈值
DefectConfig.PIXEL_AREA_BINS = DefectConfig._calculate_pixel_areas()
