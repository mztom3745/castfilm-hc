from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

from detect_core.defect_config import DefectConfig
from detect_core.castfilm_detection.api import analyze_array
from detect_core.castfilm_detection.result import DetectionResult
from config.constant import zsz_Constants

class CastFilmDefectDetector:
    """适配 CastFilm 新检测算法的包装器，保持 detect_core 接口。"""

    def __init__(self, min_component: int | None = None, blur_mode: str | None = None, profile: bool = False):
        # base_min = max(10, getattr(DefectConfig, "DETECT_AREA_MIN", 8))
        self.min_component = zsz_Constants.MIN_PIXELS
        self.blur_mode = blur_mode or "opencv"
        self.profile = profile
        self.default_left = int(getattr(DefectConfig, "LEFT_EDGE_X", 0))
        self.default_right = int(getattr(DefectConfig, "RIGHT_EDGE_X", 0))

    def set_first_image(self, _image: np.ndarray) -> None:
        """为兼容旧接口保留，无需实际操作。"""
        return

    def detect_defects_fast(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int, int, float]], int, int]:
        """
        运行 CastFilm 检测算法，返回缺陷列表与膜左右边界。
        :return: ([(x, y, w, h, area, dark_ratio), ...], left_edge, right_edge_exclusive)
        """
        gray = self._ensure_grayscale(image)
        result = analyze_array(
            gray,
            min_component=self.min_component,
            blur_mode=self.blur_mode,
            profile=self.profile,
            name="stream_frame",
        )
        self._log_anomalies(result)
        left_edge, right_edge = self._sanitize_bounds(result, gray.shape[1])
        boxes = result.boxes or []
        defects = [self._convert_box(box) for box in boxes]
        return defects, left_edge, right_edge

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("检测算法仅支持二维或三维图像数组。")

    @staticmethod
    def _convert_box(box) -> Tuple[int, int, int, int, int, float]:
        width = int(box.right - box.left + 1)
        height = int(box.bottom - box.top + 1)
        area = int(getattr(box, "pixels", width * height))
        dark_ratio = float(getattr(box, "dark_ratio", 0.0))
        return int(box.left), int(box.top), width, height, area, dark_ratio

    def _sanitize_bounds(self, result: DetectionResult, width: int) -> Tuple[int, int]:
        bounds = result.membrane_bounds
        if bounds is None:
            left, right = self.default_left, self.default_right
        else:
            left, right = bounds
        return left, right

    def _log_anomalies(self, result: DetectionResult) -> None:
        if result.membrane_issue and result.membrane_issue_reason:
            logger.warning("膜异常: {}", result.membrane_issue_reason)
        if result.detection_anomaly and result.detection_anomaly_reason:
            logger.warning("缺陷异常: {}", result.detection_anomaly_reason)
