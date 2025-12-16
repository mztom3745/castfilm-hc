from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, List

import numpy as np

from detect_core.castfilm_detection.common.filesystem import iter_image_files
from detect_core.castfilm_detection.detection_statistics import CastFilmDetectionStatistics
from detect_core.castfilm_detection.pipeline import (
    analyze_image,
    build_suggestion,
    evaluate_detection_anomaly,
    process_single_image,
    summarize_dot_classes,
)
from detect_core.castfilm_detection.result import DetectionResult
from detect_core.defect_config import DefectConfig


def analyze_image_file(
    image_path: str | Path,
    *,
    min_component: int = 8,
    save_annotation: bool = False,
    output_dir: str | Path | None = None,
    profile: bool = False,
    blur_mode: str = "integral",
) -> DetectionResult:
    """Detect membrane与缺陷并返回检测结果，可选生成标注图."""
    path = Path(image_path)
    out_dir = Path(output_dir) if output_dir is not None else None
    return process_single_image(
        path,
        output_dir=out_dir,
        min_component=min_component,
        save_annotation=save_annotation,
        collect_timings=profile,
        blur_mode=blur_mode,
    )


def analyze_array(
    gray_array: np.ndarray,
    *,
    min_component: int = 8,
    blur_mode: str = "integral",
    profile: bool = False,
    name: str = "<array>",
) -> DetectionResult:
    """Detect defects directly from an in-memory grayscale numpy array."""
    if gray_array.ndim != 2:
        raise ValueError("analyze_array expects a 2D grayscale array.")
    start_time = perf_counter()
    gray_view = np.asarray(gray_array)
    gray_float = gray_view.astype(np.float32, copy=False)
    statistics = CastFilmDetectionStatistics()
    (
        background_mean,
        threshold_value,
        boxes,
        membrane_bounds,
        timings,
        membrane_source,
        membrane_issue_reason,
    ) = analyze_image(
        gray_float,
        min_component=min_component,
        blur_mode=blur_mode,
    )
    size_distribution = statistics.summarize_defect_sizes(box.pixels for box in boxes)
    dark_threshold = float(getattr(DefectConfig, "DARK_RATIO_THRESHOLD", getattr(DefectConfig, "DARK_VALUE", 0.3)))
    min_dark_pixels = max(20, int(getattr(DefectConfig, "DARK_MIN_PIXELS", 0)))
    dot_class_counts = summarize_dot_classes(boxes, dark_threshold, min_pixels=min_dark_pixels)
    detection_anomaly, detection_anomaly_reason = evaluate_detection_anomaly(
        boxes,
        membrane_bounds,
        gray_float.shape,
    )
    detection_area = statistics.compute_detection_area(gray_float.shape, membrane_bounds)
    out_of_range, suggestion = build_suggestion(background_mean)
    elapsed = perf_counter() - start_time
    size_distribution = statistics.summarize_defect_sizes(box.pixels for box in boxes)
    return DetectionResult(
        image_name=name,
        background_mean=background_mean,
        threshold_value=threshold_value,
        boxes=boxes,
        size_distribution=size_distribution,
        membrane_bounds=membrane_bounds,
        out_of_range=out_of_range,
        suggestion=suggestion,
        defect_count=len(boxes),
        processing_time=elapsed,
        timings=timings if profile else None,
        membrane_source=membrane_source,
        membrane_issue=membrane_issue_reason is not None,
        membrane_issue_reason=membrane_issue_reason,
        detection_anomaly=detection_anomaly,
        detection_anomaly_reason=detection_anomaly_reason,
        detection_area=detection_area,
        dot_class_counts=dot_class_counts,
    )


def analyze_directory(
    input_dir: str | Path,
    *,
    min_component: int = 8,
    save_annotation: bool = False,
    output_dir: str | Path | None = None,
    profile: bool = False,
    blur_mode: str = "integral",
) -> List[DetectionResult]:
    """批量检测目录中的所有图片，返回检测结果列表."""
    input_path = Path(input_dir)
    files: Iterable[Path] = iter_image_files(input_path)
    results: List[DetectionResult] = []
    for image_path in files:
        results.append(
            analyze_image_file(
                image_path,
                min_component=min_component,
                save_annotation=save_annotation,
                output_dir=output_dir,
                profile=profile,
                blur_mode=blur_mode,
            )
        )
    return results
