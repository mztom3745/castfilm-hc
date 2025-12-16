from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from detect_core.castfilm_detection.pipeline import (
    _resolve_membrane_region,
    _compute_background_outside_membrane,
    compute_difference_map,
    BACKGROUND_RATIO,
    annotate_image,
    DARK_BG_SERIOUS,
    DARK_BG_WARN,
    DARK_BG_SCALE_SERIOUS,
    DARK_BG_SCALE_WARN,
    DARK_BG_THRESHOLD_CAP,
    DARK_RATIO_MARGIN_MIN,
    DARK_RATIO_MARGIN_RATIO,
    determine_defect_threshold,
)
from detect_core.castfilm_detection.processing.membrane import build_membrane_mask
from detect_core.castfilm_detection.processing.defects import (
    BoundingBox,
    suppress_dense_clusters,
    extract_bounding_boxes,
)
from detect_core.castfilm_detection.common.filesystem import (
    ensure_directory,
    iter_image_files,
)


def load_grayscale(image_path: Path) -> Image.Image:
    """以与 pipeline 相同的方式读成灰度图。"""
    img = Image.open(image_path)
    return img.convert("L")


def save_membrane_overlay(
    gray_img: Image.Image,
    membrane_mask: np.ndarray,
    out_path: Path,
) -> None:
    """在原图（灰度→RGB）上把膜区域高亮出来并保存。"""
    base = gray_img.convert("RGB")
    arr = np.array(base, dtype=np.uint8)  # H, W, 3
    mask = membrane_mask.astype(bool)  # H, W

    highlight_color = np.array([0, 255, 0], dtype=np.uint8)  # 绿色高亮
    arr[mask] = (
        0.5 * arr[mask].astype(np.float32) + 0.5 * highlight_color.astype(np.float32)
    ).astype(np.uint8)

    out_img = Image.fromarray(arr)
    ensure_directory(out_path.parent)
    out_img.save(out_path)


def save_difference_map_vis(
    difference_map: np.ndarray,
    out_path: Path,
) -> None:
    """将 difference_map 归一化到 [0, 255]，保存为灰度图。"""
    diff = np.maximum(difference_map, 0.0)
    max_val = float(diff.max())
    if max_val > 0:
        norm = (diff / max_val * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(diff, dtype=np.uint8)

    img = Image.fromarray(norm)
    ensure_directory(out_path.parent)
    img.save(out_path)


def save_binary_mask(
    mask: np.ndarray,
    out_path: Path,
) -> None:
    """保存二值 mask（0/255 灰度图），方便看缺陷区域。"""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(mask_uint8)
    ensure_directory(out_path.parent)
    img.save(out_path)


def save_boxes_overlay(
    gray_img: Image.Image,
    boxes: List[BoundingBox],
    membrane_bounds: Tuple[int, int] | None,
    out_path: Path,
) -> None:
    """使用 pipeline.annotate_image 在原图上画出膜边界和缺陷框。"""
    annotated = annotate_image(gray_img, boxes, membrane_bounds)
    ensure_directory(out_path.parent)
    annotated.save(out_path)


def run_pipeline_step_by_step(
    gray_array: np.ndarray,
    min_component: int,
    blur_mode: str,
) -> Tuple[
    float,                     # background_mean
    float,                     # threshold_value
    List[BoundingBox],         # boxes
    np.ndarray,                # membrane_mask
    Tuple[int, int],           # membrane_bounds
    str,                       # membrane_source
    str | None,                # membrane_issue_reason
    np.ndarray,                # difference_map (去趋势后)
    np.ndarray,                # defect_mask
]:
    """
    按 pipeline.analyze_image 的逻辑，手动展开每一步：
      1) 膜检测
      2) 背景亮度
      3) 差分图
      4) 阈值
      5) 缺陷连通域
    """
    gray_float = gray_array.astype(np.float32)

    # 1) 膜检测：build_membrane_mask + _resolve_membrane_region
    from detect_core.zsz.membrane_grad_core import build_membrane_mask_grad
    # raw_membrane_mask, detected_bounds = build_membrane_mask_grad(
    #     gray_float,
    #     ratio=BACKGROUND_RATIO,
    #     smoothing_ratio=0.003,
    #     background_percentile=0.95,
    #     drop_ratio=0.03,
    #     min_run_ratio=0.002,
    #     edge_clip_ratio=0.1,
    # )
    raw_membrane_mask, detected_bounds, background_mean = build_membrane_mask_grad(
        gray_float,
        smoothing_ratio=0.003,
        background_percentile=0.99,
        min_background_value=210,
    )
    membrane_mask, membrane_bounds, membrane_source, membrane_issue_reason = (
        _resolve_membrane_region(
            gray_float,
            raw_membrane_mask,
            detected_bounds,
        )
    )
    
    # # 2) 膜外背景亮度
    # background_mean, _ = _compute_background_outside_membrane(
    #     gray_float,
    #     membrane_bounds,
    #     membrane_source,
    # )

    # 3) 差分图（只在膜左右边界内做 diff，再填回整图），然后按列去趋势
    left, right = membrane_bounds
    print("左边：",left,"右边：",right)
    difference_map = compute_difference_map(
        gray_float,
        left,
        right,
        blur_mode=blur_mode,
    )
    col_median = np.median(difference_map, axis=0, keepdims=True)
    difference_map = np.clip(difference_map - col_median, 0, None)

    # 4) 阈值：membrane 区内的差分值 -> determine_defect_threshold
    if membrane_mask.any():
        membrane_values = difference_map[membrane_mask]
        membrane_mean = float(membrane_values.mean())
        threshold_value = determine_defect_threshold(
            membrane_values,
            membrane_mean=membrane_mean,
        )
    else:
        membrane_values = np.array([], dtype=np.float32)
        membrane_mean = None
        threshold_value = 0.0

    # 背景过暗时抬高阈值（与 pipeline 一致）
    if background_mean < DARK_BG_SERIOUS:
        threshold_value = min(
            threshold_value * DARK_BG_SCALE_SERIOUS,
            DARK_BG_THRESHOLD_CAP,
        )
    elif background_mean < DARK_BG_WARN:
        threshold_value = min(
            threshold_value * DARK_BG_SCALE_WARN,
            DARK_BG_THRESHOLD_CAP,
        )

    # 5) 缺陷连通域：difference_map + membrane_mask + 阈值
    defect_mask = (difference_map > threshold_value) & membrane_mask
    defect_mask = suppress_dense_clusters(
        defect_mask,
        tile_size=256,
        density_threshold=0.08,
    )

    dark_margin = DARK_RATIO_MARGIN_MIN
    if threshold_value is not None:
        dark_margin = max(
            DARK_RATIO_MARGIN_MIN,
            threshold_value * DARK_RATIO_MARGIN_RATIO,
        )

    boxes = extract_bounding_boxes(
        defect_mask,
        min_pixels=min_component,
        reference_map=difference_map,
        reference_threshold=threshold_value,
        reference_margin=dark_margin,
    )

    return (
        background_mean,
        threshold_value,
        boxes,
        membrane_mask,
        membrane_bounds,
        membrane_source,
        membrane_issue_reason,
        difference_map,
        defect_mask,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug visualization for CastFilm pipeline (step-by-step)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data"),  # 默认：当前目录下 data 文件夹
        help="输入图片文件夹（会递归遍历所有图片）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./debug_outputs"),  # 默认输出目录
        help="输出结果根目录。",
    )
    parser.add_argument(
        "--min-component",
        type=int,
        default=8,
        help="缺陷连通域最小像素数（用于 extract_bounding_boxes）。",
    )
    parser.add_argument(
        "--blur-mode",
        choices=("integral", "integral_shared", "opencv"),
        default="integral",
        help="与 pipeline 中一致的 blur_mode。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output
    min_component: int = args.min_component
    blur_mode: str = args.blur_mode

    ensure_directory(output_dir)
    image_files: List[Path] = list(iter_image_files(input_dir))
    if not image_files:
        print(f"[WARN] 未在目录 {input_dir} 下找到图片文件")
        return

    print(f"🔍 共找到 {len(image_files)} 张图片，将逐一处理...")
    threshold_log_lines: List[str] = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n📸 [{idx}/{len(image_files)}] 处理图像: {image_path.name}")

        # 1) 读灰度图（PIL），并转为 numpy
        gray_img = load_grayscale(image_path)
        gray_array = np.array(gray_img, dtype=np.float32)

        (
            background_mean,
            threshold_value,
            boxes,
            membrane_mask,
            membrane_bounds,
            membrane_source,
            membrane_issue_reason,
            difference_map,
            defect_mask,
        ) = run_pipeline_step_by_step(
            gray_array,
            min_component=min_component,
            blur_mode=blur_mode,
        )

        print(
            f"background_mean={background_mean:.2f}, "
            f"threshold_value={threshold_value:.3f}, "
            f"boxes={len(boxes)}, "
            f"membrane_source={membrane_source}"
        )
        if membrane_issue_reason:
            print(f"   ⚠️ membrane_issue: {membrane_issue_reason}")

        # 记录 threshold 到列表中，稍后写入 txt
        threshold_log_lines.append(
            f"{image_path.name}\tthreshold={threshold_value:.6f}\tbackground={background_mean:.3f}"
        )

        # 准备当前图像输出子目录
        img_out_dir = output_dir / image_path.stem
        ensure_directory(img_out_dir)

        # 2) 保存膜区域在原图上的高亮叠加图
        membrane_overlay_path = img_out_dir / "membrane_overlay.png"
        save_membrane_overlay(gray_img, membrane_mask, membrane_overlay_path)

        # 3) 保存 difference_map 可视化（归一化灰度图）
        diff_map_path = img_out_dir / "difference_map.png"
        save_difference_map_vis(difference_map, diff_map_path)

        # 4) 保存缺陷二值 mask
        defect_mask_path = img_out_dir / "defect_mask.png"
        save_binary_mask(defect_mask, defect_mask_path)

        # 5) 保存 boxes + 膜边界叠加在原图上的结果
        boxes_overlay_path = img_out_dir / "boxes_overlay.png"
        save_boxes_overlay(gray_img, boxes, membrane_bounds, boxes_overlay_path)

        print(
            f"   ✅ 已保存: membrane_overlay / difference_map / defect_mask / boxes_overlay 到 {img_out_dir}"
        )

        gray_img.close()  # 方便 GC

    # 写出 threshold_value 汇总 txt
    thresholds_txt_path = output_dir / "thresholds.txt"
    with thresholds_txt_path.open("w", encoding="utf-8") as f:
        f.write("image_name\tthreshold_value\tbackground_mean\n")
        for line in threshold_log_lines:
            f.write(line + "\n")

    print(f"\n📄 所有图像的 threshold_value 已写入: {thresholds_txt_path}")
    print("🎉 完成！")


if __name__ == "__main__":
    main()
