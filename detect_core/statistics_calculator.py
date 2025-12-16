from .defect_config import DefectConfig
import numpy as np
from loguru import logger

AREA_PER_PIXELS = DefectConfig.UM_PER_PIXEL * DefectConfig.UM_PER_PIXEL / 1000000000000

class StatisticsCalculator:
    def __init__(self):
        pass

    def calculate_size_index(self, area):
        """
        根据缺陷面积计算尺寸索引
        :param area: 缺陷面积（像素）
        :return: 尺寸索引
        """
        size_bins = DefectConfig.PIXEL_AREA_BINS
        size_index = int(np.digitize([area], size_bins)[0] - 1)
        size_index = max(0, min(size_index, len(DefectConfig.SIZE_LIST) - 1))
        return size_index

    def calculate_dark_area_ratio(self, defect_region):
        """
        计算缺陷区域的暗区域比例，用于判断晶点/黑点
        :param defect_region: 缺陷区域图像 (numpy.ndarray)
        :return: 暗区域比例 (float)
        """
        # 转换为灰度图像
        if defect_region.ndim == 3:
            gray_image = defect_region[:, :, 0]  # 简单取第一个通道作为灰度
        else:
            gray_image = defect_region

        # 计算统计值
        defect = np.sum(gray_image <= DefectConfig.UPPER_BOUND)
        dark_defect = np.sum(gray_image <= DefectConfig.DARK_BOUND)

        if defect > 0:
            return dark_defect / defect
        return 0.0

    def statistical_datas(self, statistical_data, image):
        width, height = image.shape
        area = width * height * AREA_PER_PIXELS ## 平方微米
        statistical_data["area_total"] += area
        ditance = (height * DefectConfig.UM_PER_PIXEL) / 1000000
        statistical_data["distance_total"] += ditance

    def determine_dot_type(self, defect_region):
        """
        根据缺陷区域判断是晶点还是黑点
        :param defect_region: 缺陷区域图像 (numpy.ndarray)
        :return: "晶点" 或 "黑点"
        """
        dark_area_ratio = self.calculate_dark_area_ratio(defect_region)
        return "晶点" if dark_area_ratio < DefectConfig.DARK_VALUE else "黑点"

    def calculate_statistics(self, defect_infos):
        """
        根据缺陷信息计算统计结果
        :param defect_infos: 缺陷信息列表，每个元素包含 {
            'defect_class': str,  # 缺陷类别
            'size_index': int,    # 尺寸索引
            'pixel_area': float,  # 像素面积
            'coord': tuple        # 坐标信息 (x, y, w, h)
        }
        :return: dict，包含：
                 {
                   'class_counts': 各类别数量统计,
                   'size_counts': 各尺寸区间数量统计,
                   'class_size_counts': 类别×尺寸 统计矩阵,
                   'boxes': 缺陷坐标列表 [(x, y, w, h), ...]
                 }
        """
        # 初始化统计结果
        total_defect_class_num = DefectConfig.DEFECT_CLASSES.copy()
        total_defect_size_num = {size: 0 for size in DefectConfig.SIZE_LIST}
        size_bin_count = len(DefectConfig.SIZE_LIST)
        total_defect_class_size_num = {k: [0] * size_bin_count for k in total_defect_class_num}
        total_defect_cord = []

        # 处理每个缺陷信息
        for info in defect_infos:
            defect_class = info['defect_class']
            size_index = info['size_index']
            coord = info['coord']

            # 添加坐标信息
            total_defect_cord.append(coord)

            # 更新类别计数
            if defect_class in total_defect_class_num:
                total_defect_class_num[defect_class] += 1
            else:
                logger.warning(f"未知的缺陷类别: {defect_class}")

            # 更新尺寸分类计数
            if 0 <= size_index < len(DefectConfig.SIZE_LIST):
                total_defect_size_num[DefectConfig.SIZE_LIST[size_index]] += 1

                # 更新缺陷类别和尺寸分类计数
                if defect_class in total_defect_class_size_num:
                    total_defect_class_size_num[defect_class][size_index] += 1
            else:
                logger.warning(f"无效的尺寸索引: {size_index}")

        return {
            'class_counts': total_defect_class_num,
            'size_counts': total_defect_size_num,
            'class_size_counts': total_defect_class_size_num,
            'boxes': total_defect_cord
        }