from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from PIL import Image, ImageDraw

from detect_core.castfilm_detection.common.filesystem import ensure_directory, iter_image_files, write_report
from detect_core.castfilm_detection.detection_statistics import CastFilmDetectionStatistics
from detect_core.castfilm_detection.processing.defects import BoundingBox, extract_bounding_boxes, suppress_dense_clusters
from detect_core.castfilm_detection.processing.filters import box_blur, box_blur_opencv, box_blur_pair_shared
from detect_core.castfilm_detection.processing.membrane import build_membrane_mask
from detect_core.castfilm_detection.result import DetectionResult
from detect_core.defect_config import DefectConfig
from detect_core.zsz import membrane_grad_core

from config.constant import zsz_Constants
BLUR_MODES = ("integral", "integral_shared", "opencv")
POOL_MODES = ("thread", "process")
_FALLBACK_WARNED = False
BACKGROUND_RATIO = 0.1
BACKGROUND_RANGE = (215.0, 225.0)
REPORT_FILENAME = "detection_report.txt"
DEFAULT_MEMBRANE_MARGIN_RATIO = 0.2
MAX_MEMBRANE_MARGIN_RATIO = DEFAULT_MEMBRANE_MARGIN_RATIO
MIN_MEMBRANE_WIDTH_RATIO = 0.4
ABNORMAL_COVERAGE_RATIO = 0.05
ABNORMAL_DEFECT_COUNT = 300
ABNORMAL_MAX_BOX_RATIO = 0.02
DARK_BG_WARN = 180.0
DARK_BG_SERIOUS = 150.0
DARK_BG_SCALE_WARN = 1.5
DARK_BG_SCALE_SERIOUS = 2.0
DARK_BG_THRESHOLD_CAP = 120.0
DARK_RATIO_MARGIN_MIN = 10.0
DARK_RATIO_MARGIN_RATIO = 0.25
FIXED_BACKGROUND = 220.0

_LAST_MEMBRANE_BOUNDS: Optional[Tuple[int, int]] = None
_LAST_MEMBRANE_WIDTH: Optional[int] = None


def _default_membrane_bounds(width: int) -> Tuple[int, int]:
    margin = max(1, int(width * DEFAULT_MEMBRANE_MARGIN_RATIO))
    if margin * 2 >= width:
        margin = max(1, width // 4)
    left = margin
    right = max(left + 1, width - margin - 1)
    return left, right


def _membrane_bounds_reasonable(bounds: Optional[Tuple[int, int]], width: int) -> bool:
    if bounds is None:
        return False
    left, right = bounds
    if left < 0 or right >= width or right <= left:
        return False
    max_margin = int(width * MAX_MEMBRANE_MARGIN_RATIO)
    left_margin = left
    right_margin = width - 1 - right
    if left_margin > max_margin or right_margin > max_margin:
        return False
    min_width = max(2, int(width * MIN_MEMBRANE_WIDTH_RATIO))
    if (right - left + 1) < min_width:
        return False
    return True


def _membrane_mask_from_bounds(shape: Tuple[int, int], bounds: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    left, right = bounds
    left = max(0, min(left, width - 2))
    right = max(left + 1, min(right, width - 1))
    mask = np.zeros((height, width), dtype=bool)
    mask[:, left : right + 1] = True
    return mask


def _previous_membrane_bounds(width: int) -> Optional[Tuple[int, int]]:
    if _LAST_MEMBRANE_WIDTH == width:
        return _LAST_MEMBRANE_BOUNDS
    return None


def _remember_membrane_bounds(bounds: Tuple[int, int], width: int) -> None:
    global _LAST_MEMBRANE_BOUNDS, _LAST_MEMBRANE_WIDTH
    _LAST_MEMBRANE_BOUNDS = bounds
    _LAST_MEMBRANE_WIDTH = width


def _compute_background_outside_membrane(
    gray: np.ndarray,
    bounds: Tuple[int, int],
    source: str,
    fixed_background: float = FIXED_BACKGROUND,
) -> Tuple[float, bool]:
    """
    Estimate background mean using membrane外侧区域；若无有效背景或使用默认边界，返回固定值。
    返回 (background_mean, used_fallback)
    """
    # [背景左侧区域] | [膜区域]  | [背景右侧区域]
    # band_left     | membrane | band_right
    
    if source == "default":
        return fixed_background, True
    height, width = gray.shape
    left, right = bounds
    left = max(0, min(int(left), width - 1))
    right = max(left, min(int(right), width - 1))
    bands = []
    if left > 0:
        band_left = gray[:, :left]
        if band_left.size:
            bands.append(band_left)
    if right < width - 1:
        band_right = gray[:, right + 1 :]
        if band_right.size:
            bands.append(band_right)
    if not bands:
        return fixed_background, True
    data = np.concatenate([band.reshape(-1) for band in bands])
    if data.size == 0:
        return fixed_background, True
    return float(data.mean()), False


def _resolve_membrane_region(
    gray: np.ndarray,
    raw_mask: np.ndarray,
    detected_bounds: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int], str, Optional[str]]:
    width = gray.shape[1]
    if _membrane_bounds_reasonable(detected_bounds, width):
        #检查这组边界靠不靠谱：
        # print("边界靠谱")
        _remember_membrane_bounds(detected_bounds, width)
        return raw_mask, detected_bounds, "detected", None
    previous = _previous_membrane_bounds(width)
    if previous is not None:
        mask = _membrane_mask_from_bounds(gray.shape, previous)
        return mask, previous, "previous", "膜边界定位失败，已沿用上一帧边界"
    fallback_bounds = _default_membrane_bounds(width)
    mask = _membrane_mask_from_bounds(gray.shape, fallback_bounds)
    return mask, fallback_bounds, "default", "膜边界定位失败，已使用默认 20% 边界"


def evaluate_detection_anomaly(
    boxes: List[BoundingBox],
    bounds: Tuple[int, int],
    image_shape: Tuple[int, int],
) -> Tuple[bool, Optional[str]]:
    height, _ = image_shape
    roi_width = max(1, bounds[1] - bounds[0] + 1)
    roi_area = roi_width * max(1, height)

    def area(box: BoundingBox) -> int:
        return (box.right - box.left + 1) * (box.bottom - box.top + 1)

    total_area = sum(area(box) for box in boxes)
    max_area = max((area(box) for box in boxes), default=0)
    coverage = total_area / roi_area
    max_area_ratio = max_area / roi_area

    if coverage >= ABNORMAL_COVERAGE_RATIO:
        return True, f"缺陷覆盖率异常（{coverage:.1%}）"
    if len(boxes) >= ABNORMAL_DEFECT_COUNT:
        return True, f"缺陷数量异常（{len(boxes)}）"
    if max_area_ratio >= ABNORMAL_MAX_BOX_RATIO:
        return True, f"单个缺陷面积异常（{max_area_ratio:.1%}）"
    return False, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CastFilm image inspection pipeline")
    parser.add_argument("--input", type=Path, default=Path("data"), help="Input image directory")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--min-component", type=int, default=8, help="Minimum defect pixel count")
    parser.add_argument("--profile", action="store_true", help="Print per-stage timing information")
    parser.add_argument(
        "--blur-mode",
        choices=BLUR_MODES,
        default="opencv",
        help="Box blur backend used in the difference map stage.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (per-image multi-threading). Use >1 to enable concurrency.",
    )
    parser.add_argument(
        "--pool",
        choices=POOL_MODES,
        default="thread",
        help="Executor kind when workers>1.",
    )
    return parser.parse_args()

def load_grayscale(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    return image.convert("L")


def _difference_map_core(region: np.ndarray, blur_mode: str = "integral") -> np.ndarray:
    """
    Multi-scale difference map (3 scales):
      - blur sizes = (17, 61, 171)
      - diff = blur - region
      - combined = max(diff17, diff61, diff171, 0)

    Parameters
    ----------
    region : np.ndarray
        2D float array (H, W), grayscale ROI.
    blur_mode : str
        One of: "integral", "integral_shared", "opencv"

    Returns
    -------
    np.ndarray
        2D float array (H, W), non-negative combined difference map.
    """
    global _FALLBACK_WARNED

    region = np.asarray(region, dtype=np.float32)
    sizes = (17, 61, 171)

    if blur_mode == "opencv":
        try:
            blur17 = box_blur_opencv(region, size=sizes[0])
            blur61 = box_blur_opencv(region, size=sizes[1])
            blur171 = box_blur_opencv(region, size=sizes[2])
        except RuntimeError:
            # 不打印，只 fallback
            _FALLBACK_WARNED = True
            return _difference_map_core(region, blur_mode="integral_shared")

    elif blur_mode == "integral_shared":
        blur_map = box_blur_pair_shared(region, sizes)
        blur17 = blur_map[sizes[0]]
        blur61 = blur_map[sizes[1]]
        blur171 = blur_map[sizes[2]]

    else:  # "integral"
        blur17 = box_blur(region, size=sizes[0])
        blur61 = box_blur(region, size=sizes[1])
        blur171 = box_blur(region, size=sizes[2])

    diff17 = blur17 - region
    diff61 = blur61 - region
    diff171 = blur171 - region

    combined = np.maximum.reduce([diff17, diff61, diff171, np.zeros_like(region)])
    return combined

def compute_difference_map(
    gray_float: np.ndarray,
    left: int,
    right: int,
    blur_mode: str = "integral",
) -> np.ndarray:
    """
    Clean version:
      1) 在 ROI 计算三尺度 combined diff
      2) ROI 内按行减中位数（去行趋势）
      3) 全图按列减中位数（去列趋势）
      4) 全程 clip 到 >=0
    """
    gray_float = np.asarray(gray_float, dtype=np.float32)
    h, w = gray_float.shape

    left = max(0, int(left))
    right = min(w - 1, int(right))
    if right <= left:
        raise ValueError("ROI bounds are invalid for difference computation.")

    roi = gray_float[:, left : right + 1]  # (H, ROI_W)
    diff_roi = _difference_map_core(roi, blur_mode=blur_mode)

    # 1) 行中位数（只在 ROI 上算）
    row_median = np.median(diff_roi, axis=1, keepdims=True).astype(np.float32)  # (H,1)

    # 2) 填回整图
    diff_full = np.zeros((h, w), dtype=np.float32)
    diff_full[:, left : right + 1] = diff_roi

    # 3) 去行趋势 + clip
    diff_row = np.clip(diff_full - row_median, 0.0, None)

    # 4) 去列趋势 + clip
    col_median = np.median(diff_row, axis=0, keepdims=True).astype(np.float32)  # (1,W)
    diff_row_col = np.clip(diff_row - col_median, 0.0, None)

    return diff_row_col


def determine_defect_threshold(values: np.ndarray, membrane_mean: float | None = None) -> float:
    # values = difference_map[membrane_mask] ——膜内的全部差分值
    flat = values.ravel().astype(np.float32)
    if flat.size == 0:
        return 0.0
    positives = flat[flat > 0]
    if positives.size == 0:
        positives = flat
    max_samples = 500_000
    sample = positives
    sample_ratio = 0.1
    if sample.size > max_samples:
        stride = math.ceil(sample.size / max_samples)
        sample = sample[::stride]
    elif sample_ratio < 1.0:
        stride = max(1, int(1.0 / sample_ratio))
        sample = sample[::stride]
    # 如果膜区像素太多（几百万），做下采样到 10% 或最多 50万。    
    mean_value = float(sample.mean())
    std_value = float(sample.std())
    median_value = float(np.median(sample))
    mad_value = float(np.median(np.abs(sample - median_value)))
    percentile_95 = float(np.percentile(sample, 95))
    percentile_98 = float(np.percentile(sample, 98))
    #95、98 分位
    mean_std = mean_value + 3.0 * std_value
    #均值 + 3σ
    robust = median_value + 6.0 * (1.4826 * mad_value if mad_value > 0 else 0.0)
    #中位数 + 6*MAD
    base = max(10.0, percentile_95, percentile_98, mean_std, robust)
    # 根据膜内灰度动态缩放阈值：暗膜抬高阈值，亮膜保持
    scale = 1.0
    if membrane_mean is not None:
        if membrane_mean < 190:
            scale = 1.6
        elif membrane_mean < 210:
            scale = 1.3
    return base * scale

def determine_defect_threshold(
    values: np.ndarray,
) -> Tuple[float, float]:
    """
    根据 difference_map 中的正值分布，计算：
      - threshold       : 缺陷响应阈值
      - dark_margin     : 黑点判定用的 reference_margin

    返回
    ----
    threshold : float
    dark_margin : float
    """
    flat = values.ravel().astype(np.float32)
    if flat.size == 0:
        return 0.0, 0.0

    # 只使用正响应
    positives = flat[flat > 0]
    if positives.size == 0:
        positives = flat

    # 下采样，防止超大图统计过慢
    max_samples = 500_000
    if positives.size > max_samples:
        stride = math.ceil(positives.size / max_samples)
        sample = positives[::stride]
    else:
        sample = positives

    # === 基础统计量 ===
    mean_value = float(sample.mean())
    std_value = float(sample.std())
    median_value = float(np.median(sample))
    mad_value = float(np.median(np.abs(sample - median_value)))
    # percentile_85 = float(np.percentile(sample, 85))

    # === 缺陷阈值（保持你当前逻辑）===
    base = max(
        zsz_Constants.MIN_GRAY, 
        min(mean_value, median_value, zsz_Constants.MAX_GRAY)
    )

    threshold = base

    # === 黑点 margin（用于 dark_ratio / reference_map）===
    black_threshold1 = mean_value + 2.0 * std_value
    black_threshold2 = median_value + 6.0 * (1.4826 * mad_value if mad_value > 0 else 0.0)

    dark_margin = min(
        zsz_Constants.MAX_DARK,
        max(zsz_Constants.MIN_DARK, black_threshold1, black_threshold2),
    )

    return threshold, dark_margin

def summarize_dot_classes(boxes: List[BoundingBox], dark_ratio: float, min_pixels: int = 0) -> Dict[str, int]:
    """Count 黑点/晶点 based on dark_ratio and threshold;可设最小像素数过滤。"""
    counts = {"晶点": 0, "黑点": 0}
    for box in boxes:
        #如果小于min_pixels，就看成晶点
        if min_pixels and getattr(box, "pixels", 0) < min_pixels:
            counts["晶点"] += 1
            continue
        #如果比例大于dark_ratio，就看成黑点
        ratio = float(getattr(box, "dark_ratio", 0.0))
        if ratio >= dark_ratio:
            counts["黑点"] += 1
        else:
            counts["晶点"] += 1
    return counts


def analyze_image(
    gray_array: np.ndarray,
    min_component: int,
    blur_mode: str = "integral",
) -> Tuple[
    float,
    float,
    List[BoundingBox],
    Tuple[int, int],
    Dict[str, float],
    str,
    Optional[str],
]:  
    """
    这里简化膜检测：整幅图视为膜区域。
    按步骤执行：
      Step 1: 膜区域/背景
      Step 2: 差分图计算（含 _difference_map_core）
      Step 3: 阈值计算（determine_defect_threshold）
      Step 4: 原始缺陷 mask
      Step 5: suppress_dense_clusters
      Step 6: extract_bounding_boxes
    """
    timings: Dict[str, float] = {}

    gray_float = gray_array.astype(np.float32)
    from detect_core.zsz.membrane_grad_core import build_membrane_mask_grad
    t0 = perf_counter()
    membrane_mask, membrane_bounds, background_mean = build_membrane_mask_grad(
        gray_float,
        smoothing_ratio=0.003,
        background_percentile=0.99,
        min_background_value=210,
    )


    timings["membrane_detection"] = perf_counter() - t0
    #新增，裁剪膜宽至8cm,计算方式是
    from config.constant import Constants
    pix_w = math.ceil((80000 / Constants.UM_PER_PIXEL))
    left, right = membrane_bounds
    current_w = right - left + 1
    if current_w > pix_w:
        # 取中间的 pix_w 宽度
        center = (left + right) // 2
        half = pix_w // 2

        new_left = max(0, center - half)
        new_right = new_left + pix_w - 1

        # 防止越界
        new_right = min(new_right, gray_float.shape[1] - 1)
        new_left = new_right - pix_w + 1

        membrane_bounds = (new_left, new_right)

    # (
    #     membrane_mask,
    #     membrane_bounds,
    #     membrane_source,
    #     membrane_issue_reason,
    # ) = _resolve_membrane_region(
    #     gray_float,
    #     membrane_mask,
    #     membrane_bounds,
    # )
    membrane_source, membrane_issue_reason = "detected", None
    #强制8cm检测中
    
    # 2) difference_map
    t0 = perf_counter()
    difference_map = compute_difference_map(
        gray_float,
        membrane_bounds[0],
        membrane_bounds[1],
        blur_mode=blur_mode,
    )
    #只取了缺陷
    # 新增：把所有 < 15 的值强制设为 0
    difference_map = np.where(difference_map < zsz_Constants.NOICE_GRAY, 0, difference_map)

    timings["difference_map"] = perf_counter() - t0

    # 3) 阈值（只看膜内像素，此处即全图）
    # dark_margin返回暗点值
    t0 = perf_counter()
    membrane_values = difference_map[membrane_mask]
    threshold_value, dark_margin = determine_defect_threshold(
        membrane_values,
    )
    
    #背景低于210的地方不会再被计算为背景
    # if background_mean < DARK_BG_SERIOUS:
    #     threshold_value = min(threshold_value * DARK_BG_SCALE_SERIOUS, DARK_BG_THRESHOLD_CAP)
    # elif background_mean < DARK_BG_WARN:
    #     threshold_value = min(threshold_value * DARK_BG_SCALE_WARN, DARK_BG_THRESHOLD_CAP)
    
    timings["threshold"] = perf_counter() - t0

    # 4) defect_mask（原始） = difference_map > threshold 且在膜内
    # suppress_dense_clusters
    t0 = perf_counter()
    defect_mask = (difference_map > threshold_value) & membrane_mask
    defect_mask = suppress_dense_clusters(
        defect_mask, tile_size=256, density_threshold=0.08
    )

    # if threshold_value is not None:
    #     dark_margin = max(DARK_RATIO_MARGIN_MIN, threshold_value * DARK_RATIO_MARGIN_RATIO)

    boxes = extract_bounding_boxes(
        defect_mask,
        min_pixels=min_component,
        reference_map=difference_map,
        reference_threshold=threshold_value,
        reference_margin=dark_margin,
    )
    timings["defect_extraction"] = perf_counter() - t0
    
    return (
        background_mean,
        threshold_value,
        boxes,
        membrane_bounds,
        timings,
        membrane_source,
        membrane_issue_reason,
    )


def annotate_image(original: Image.Image, boxes: List[BoundingBox], membrane_bounds: tuple[int, int] | None = None) -> Image.Image:
    annotated = original.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    height = annotated.height
    if membrane_bounds is not None:
        left_bound, right_bound = membrane_bounds
        draw.line([(left_bound, 0), (left_bound, height)], fill=(0, 255, 0), width=2)
        draw.line([(right_bound, 0), (right_bound, height)], fill=(0, 255, 0), width=2)
    for box in boxes:
        left, top, right, bottom = box.left, box.top, box.right, box.bottom
        draw.rectangle(
            [(left, top), (right + 1, bottom + 1)],
            outline=(255, 0, 0),
            width=2,
        )
    return annotated


def save_defect_crops(original: Image.Image, boxes: List[BoundingBox], crops_dir: Path, crop_size: int = 64) -> None:
    """Save each detected defect as a fixed-size crop (defect at center) for manual inspection."""
    if not boxes:
        return
    ensure_directory(crops_dir)
    for idx, box in enumerate(boxes):
        cx = int(round((box.left + box.right) / 2))
        cy = int(round((box.top + box.bottom) / 2))
        half = crop_size // 2
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(original.width, left + crop_size)
        bottom = min(original.height, top + crop_size)
        # 如果靠边导致尺寸不足，向回补齐窗口
        if right - left < crop_size and left > 0:
            left = max(0, right - crop_size)
        if bottom - top < crop_size and top > 0:
            top = max(0, bottom - crop_size)
        crop = original.crop((left, top, right, bottom))
        crop.save(crops_dir / f"defect_{idx:04d}.png")


def build_suggestion(background_mean: float) -> tuple[bool, str | None]:
    low, high = BACKGROUND_RANGE
    if background_mean < low:
        suggestion = (
            f"Background too dark; increase illumination (current {background_mean:.2f}, target {low:.0f}-{high:.0f})."
        )
        return True, suggestion
    if background_mean > high:
        suggestion = (
            f"Background too bright; decrease illumination (current {background_mean:.2f}, target {low:.0f}-{high:.0f})."
        )
        return True, suggestion
    return False, None

def format_report_entry(result: DetectionResult) -> str:
    lines = [
        f"Image: {result.image_name}",
        f"Background mean: {result.background_mean:.2f}",
    ]
    if result.out_of_range and result.suggestion:
        lines.append(f"Lighting within range: No ({result.suggestion})")
    else:
        lines.append("Lighting within range: Yes")
    lines.append(f"Defect count: {result.defect_count}")
    lines.append(f"Membrane source: {result.membrane_source}")
    if result.membrane_issue and result.membrane_issue_reason:
        lines.append(f"Membrane anomaly: Yes ({result.membrane_issue_reason})")
    else:
        lines.append("Membrane anomaly: No")
    if result.detection_anomaly and result.detection_anomaly_reason:
        lines.append(f"Detection anomaly: Yes ({result.detection_anomaly_reason})")
    else:
        lines.append("Detection anomaly: No")
    lines.append(f"Processing time: {result.processing_time:.2f} s")
    if result.size_distribution:
        lines.append("Defect size buckets:")
        stats = CastFilmDetectionStatistics()
        for label in stats.size_labels:
            count = result.size_distribution.get(label, 0)
            lines.append(f"  {label}: {count}")
    if result.dot_class_counts:
        black = result.dot_class_counts.get("黑点", 0)
        crystal = result.dot_class_counts.get("晶点", 0)
        lines.append(f"Dot classes: 黑点 {black}, 晶点 {crystal}")
    if result.timings:
        for key, value in result.timings.items():
            lines.append(f"{key} time: {value:.2f} s")

    lines.append("-" * 40)
    return "\n".join(lines)

def process_single_image(
    image_path: Path,
    output_dir: Path | None,
    min_component: int,
    save_annotation: bool = True,
    collect_timings: bool = False,
    blur_mode: str = "integral",
) -> DetectionResult:
    start_time = perf_counter()
    statistics = CastFilmDetectionStatistics()
    grayscale_image = load_grayscale(image_path)
    detection_anomaly = False
    detection_anomaly_reason: Optional[str] = None
    membrane_source = "detected"
    membrane_issue_reason: Optional[str] = None
    boxes: List[BoundingBox] = []
    size_distribution: Dict[str, int] | None = None
    detection_area: Dict[str, float] | None = None
    dot_class_counts: Dict[str, int] | None = None
    try:
        gray_array = np.array(grayscale_image, dtype=np.float32)
        (
            background_mean,
            threshold_value,
            boxes,
            membrane_bounds,
            timings,
            membrane_source,
            membrane_issue_reason,
        ) = analyze_image(
            gray_array,
            min_component=min_component,
            blur_mode=blur_mode,
        )
        detection_anomaly, detection_anomaly_reason = evaluate_detection_anomaly(
            boxes,
            membrane_bounds,
            gray_array.shape,
        )
        filtered_boxes = boxes  # 保留异常帧的框用于人工判定
        size_distribution = statistics.summarize_defect_sizes(box.pixels for box in filtered_boxes)
        detection_area = statistics.compute_detection_area(gray_array.shape, membrane_bounds)
        from config.constant import zsz_Constants
        dark_ratio = float(getattr(zsz_Constants, "DARK_RATIO", 0.3))
        min_dark_pixels = int(getattr(zsz_Constants, "MIN_PIXELS", 0))
        dot_class_counts = summarize_dot_classes(filtered_boxes, dark_ratio, min_pixels=min_dark_pixels)
        if save_annotation:
            if output_dir is None:
                raise ValueError("save_annotation=True requires output_dir to be provided.")
            annotated = annotate_image(grayscale_image, filtered_boxes, membrane_bounds)
            ensure_directory(output_dir)
            annotated.save(output_dir / image_path.name)
            annotated.close()
            # 保存缺陷裁剪图，便于人工复核
            crops_dir = output_dir / "crops" / image_path.stem
            save_defect_crops(grayscale_image, filtered_boxes, crops_dir)
    finally:
        grayscale_image.close()
    out_of_range, suggestion = build_suggestion(background_mean)
    elapsed = perf_counter() - start_time
    return DetectionResult(
        image_name=image_path.name,
        background_mean=background_mean,
        out_of_range=out_of_range,
        suggestion=suggestion,
        defect_count=len(filtered_boxes),
        processing_time=elapsed,
        timings=timings if collect_timings else None,
        threshold_value=threshold_value,
        boxes=None,
        size_distribution=size_distribution,
        membrane_bounds=membrane_bounds,
        membrane_source=membrane_source,
        membrane_issue=membrane_issue_reason is not None,
        membrane_issue_reason=membrane_issue_reason,
        detection_anomaly=detection_anomaly,
        detection_anomaly_reason=detection_anomaly_reason,
        detection_area=detection_area,
        dot_class_counts=dot_class_counts,
    )

def main() -> None:
    args = parse_args()
    image_files = list(iter_image_files(args.input))
    ensure_directory(args.output)

    def log_success(path: Path, result: DetectionResult) -> None:
        warn = result.membrane_issue or result.detection_anomaly
        label = "[WARN]" if warn else "[OK]"
        print(
            f"{label} {path.name}: background {result.background_mean:.2f}, defects {result.defect_count}, "
            f"time {result.processing_time:.2f}s"
        )
        if result.membrane_issue and result.membrane_issue_reason:
            print(f"       membrane: {result.membrane_issue_reason}")
        if result.detection_anomaly and result.detection_anomaly_reason:
            print(f"       detection: {result.detection_anomaly_reason}")
        if args.profile and result.timings:
            detail = ", ".join(f"{key} {value:.2f}s" for key, value in result.timings.items())
            print(f"       stage timings: {detail}")

    results_map: Dict[Path, DetectionResult] = {}

    if args.workers <= 1:
        for image_path in image_files:
            try:
                result = process_single_image(
                    image_path,
                    args.output,
                    args.min_component,
                    save_annotation=True,
                    collect_timings=args.profile,
                    blur_mode=args.blur_mode,
                )
                results_map[image_path] = result
                log_success(image_path, result)
            except Exception as exc:
                print(f"[ERROR] {image_path.name}: {exc}")
    else:
        workers = max(1, args.workers)
        executor_cls = ThreadPoolExecutor if args.pool == "thread" else ProcessPoolExecutor
        with executor_cls(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    process_single_image,
                    image_path,
                    args.output,
                    args.min_component,
                    save_annotation=True,
                    collect_timings=args.profile,
                    blur_mode=args.blur_mode,
                ): image_path
                for image_path in image_files
            }
            for future in as_completed(future_map):
                image_path = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[ERROR] {image_path.name}: {exc}")
                else:
                    results_map[image_path] = result
                    log_success(image_path, result)

    ordered_results = [results_map[path] for path in image_files if path in results_map]
    if ordered_results:
        report_entries = [format_report_entry(result) for result in ordered_results]
        report_path = args.output / REPORT_FILENAME
        write_report(report_path, report_entries)
        print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
