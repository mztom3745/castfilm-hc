from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from loguru import logger

MICRON_PER_METER = 1_000_000.0

try:
    from detect_core.defect_config import DefectConfig

    _DEFAULT_PIXEL_SIZE_UM = getattr(DefectConfig, "UM_PER_PIXEL", 7.0)
except Exception:
    _DEFAULT_PIXEL_SIZE_UM = 7.0


class CastFilmDetectionStatistics:
    """
    Compute membrane-ROI area and defect size stats aligned with DefectConfig thresholds.
    提供与 StatisticsCalculator 兼容的接口，便于替换老统计实现。
    """

    def __init__(self, pixel_size_um: float | None = None) -> None:
        self.pixel_size_um = float(pixel_size_um if pixel_size_um is not None else _DEFAULT_PIXEL_SIZE_UM)
        self.pixel_size_m = self.pixel_size_um / MICRON_PER_METER
        self.area_per_pixel_m2 = self.pixel_size_m * self.pixel_size_m

    # --- 面积相关 ---
    def compute_detection_area(self, image_shape: Tuple[int, int], membrane_bounds: Tuple[int, int] | None) -> Dict[str, float]:
        height, width = int(image_shape[0]), int(image_shape[1])
        if membrane_bounds is None:
            left, right = 0, width - 1
        else:
            left, right = membrane_bounds
            left = max(0, min(int(left), width - 1))
            right = max(left, min(int(right), width - 1))
        roi_width = right - left + 1
        area_pixels = roi_width * height
        length_m = height * self.pixel_size_m
        return {
            "roi_width_px": roi_width,
            "height_px": height,
            "area_pixels": area_pixels,
            "area_m2": area_pixels * self.area_per_pixel_m2,
            "length_m": length_m,
            "left_bound": left,
            "right_bound": right,
        }

    def update_area_totals(
        self,
        accumulator: Dict[str, Any],
        image_shape: Tuple[int, int],
        membrane_bounds: Tuple[int, int] | None,
    ) -> Dict[str, float]:
        area_info = self.compute_detection_area(image_shape, membrane_bounds)
        accumulator.setdefault("area_total", 0.0)
        accumulator.setdefault("distance_total", 0.0)
        accumulator["area_total"] += area_info["area_m2"]
        accumulator["distance_total"] += area_info["length_m"]
        accumulator["last_detection_area"] = area_info
        return area_info

    def statistical_datas(self, statistical_data: Dict[str, Any], image: np.ndarray) -> None:
        """兼容旧接口：直接以输入图像的宽高累计面积/距离。"""
        height, width = image.shape[:2]
        area_pixels = width * height
        area_m2 = area_pixels * self.area_per_pixel_m2
        distance_m = height * self.pixel_size_m
        statistical_data.setdefault("area_total", 0.0)
        statistical_data.setdefault("distance_total", 0.0)
        statistical_data["area_total"] += area_m2
        statistical_data["distance_total"] += distance_m
        statistical_data["last_detection_area"] = {
            "roi_width_px": width,
            "height_px": height,
            "area_pixels": area_pixels,
            "area_m2": area_m2,
            "length_m": distance_m,
            "left_bound": 0,
            "right_bound": width - 1,
        }

    # --- 尺寸相关 ---
    @property
    def size_labels(self) -> Tuple[str, ...]:
        try:
            return tuple(DefectConfig.SIZE_LIST)
        except Exception:
            return tuple()

    @property
    def size_bins(self) -> Tuple[float, ...]:
        try:
            return tuple(float(v) for v in DefectConfig.PIXEL_AREA_BINS)
        except Exception:
            return tuple()

    def compute_size_index(self, area_pixels: float) -> int:
        """Map defect pixel area to size index using DefectConfig bins."""
        bins = self.size_bins
        labels = self.size_labels
        if not bins or not labels:
            return 0
        for idx, boundary in enumerate(bins):
            if area_pixels < boundary:
                return max(0, idx - 1)
        return len(labels) - 1

    # 兼容旧命名
    def calculate_size_index(self, area: float) -> int:
        return self.compute_size_index(area)

    def summarize_defect_sizes(self, areas: Iterable[int]) -> Dict[str, int]:
        """Count defects per size bucket (labels come from DefectConfig.SIZE_LIST)."""
        labels = self.size_labels
        summary: Dict[str, int] = {label: 0 for label in labels}
        for area in areas:
            index = self.compute_size_index(float(area))
            if 0 <= index < len(labels):
                summary[labels[index]] += 1
        return summary

    # --- 晶点/黑点兼容逻辑 ---
    def calculate_dark_area_ratio(self, defect_region: np.ndarray) -> float:
        if defect_region.ndim == 3:
            gray_image = defect_region[:, :, 0]
        else:
            gray_image = defect_region
        defect = np.sum(gray_image <= DefectConfig.UPPER_BOUND)
        dark_defect = np.sum(gray_image <= DefectConfig.DARK_BOUND)
        if defect > 0:
            return dark_defect / defect
        return 0.0

    def determine_dot_type(self, defect_region: np.ndarray) -> str:
        dark_area_ratio = self.calculate_dark_area_ratio(defect_region)
        threshold = float(getattr(DefectConfig, "DARK_RATIO_THRESHOLD", DefectConfig.DARK_VALUE))
        return "晶点" if dark_area_ratio < threshold else "黑点"

    def calculate_statistics(self, defect_infos: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        total_defect_class_num = DefectConfig.DEFECT_CLASSES.copy()
        total_defect_size_num = {size: 0 for size in DefectConfig.SIZE_LIST}
        size_bin_count = len(DefectConfig.SIZE_LIST)
        total_defect_class_size_num = {k: [0] * size_bin_count for k in total_defect_class_num}
        total_defect_cord = []

        for info in defect_infos:
            defect_class = info.get("defect_class")
            size_index = info.get("size_index")
            coord = info.get("coord")

            if coord is not None:
                total_defect_cord.append(coord)

            if defect_class in total_defect_class_num:
                total_defect_class_num[defect_class] += 1
            else:
                logger.warning(f"未知的缺陷类别 {defect_class}")

            if isinstance(size_index, int) and 0 <= size_index < len(DefectConfig.SIZE_LIST):
                total_defect_size_num[DefectConfig.SIZE_LIST[size_index]] += 1
                if defect_class in total_defect_class_size_num:
                    total_defect_class_size_num[defect_class][size_index] += 1
            else:
                logger.warning(f"无效的尺寸索引 {size_index}")

        return {
            "class_counts": total_defect_class_num,
            "size_counts": total_defect_size_num,
            "class_size_counts": total_defect_class_size_num,
            "boxes": total_defect_cord,
        }
