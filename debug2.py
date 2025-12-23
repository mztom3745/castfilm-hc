from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
from PIL import Image, ImageDraw

# ========= 可复用的简单文件工具 =========

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            yield p


# ========= 从你项目里导入核心算法 =========
from detect_core.castfilm_detection.processing.filters import (
    box_blur,
    box_blur_opencv,
    box_blur_pair_shared,
)
from detect_core.castfilm_detection.processing.defects import (
    BoundingBox,
    suppress_dense_clusters,
    extract_bounding_boxes,
)

# ========= 一些全局超参数（与 pipeline 保持一致） =========

_FALLBACK_WARNED = False
BLUR_MODES = ("integral", "integral_shared", "opencv")

DARK_BG_WARN = 180.0
DARK_BG_SERIOUS = 150.0
DARK_BG_SCALE_WARN = 1.5
DARK_BG_SCALE_SERIOUS = 2.0
DARK_BG_THRESHOLD_CAP = 120.0

DARK_RATIO_MARGIN_MIN = 95.0
DARK_RATIO_MARGIN_RATIO = 0.2


# ========= 可视化辅助函数 =========

def load_grayscale(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    return img.convert("L")


def _save_float_array_as_gray(
    a: np.ndarray,
    out_path: Path,
) -> None:
    """
    将 float 图像反色（255 - 值），
    小于等于 1 的值将自然变成接近白色。
    """
    ensure_directory(out_path.parent)

    arr = np.asarray(a, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    # 反色：大值变黑，小值变白
    out = 255.0 - arr
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    Image.fromarray(out).save(out_path)

def _save_float_array_as_gray(
    a: np.ndarray,
    out_path: Path,
) -> None:
    """
    将 float 图像反色（255 - 值），
    小于等于 1 的值将自然变成接近白色。
    """
    ensure_directory(out_path.parent)

    arr = np.asarray(a, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    # 反色：大值变黑，小值变白
    out = 255.0 - arr
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    Image.fromarray(out).save(out_path)

def save_float_as_gray_raw(a: np.ndarray, out_path: Path) -> None:
    """
    完全保留原 float 数值，只做最基础的：
      - nan → 0
      - inf → 255
      - clip 到 [0,255]
      - 转 uint8
    不反色、不拉伸、不做任何增强。
    """
    ensure_directory(out_path.parent)

    arr = np.asarray(a, dtype=np.float32)
    
    # 防止 nan/inf 出问题
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    # 保持原值，直接裁剪到 [0,255]
    out = np.clip(arr, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_path)

def _save_bool_mask_as_gray(
    mask: np.ndarray,
    out_path: Path,
    invert: bool = False,
) -> None:
    """
    将 bool mask 保存为灰度图。
    
    参数
    ----
    invert : 是否黑白反转（True = 反转）。
    """
    ensure_directory(out_path.parent)

    # 转 uint8
    mask_u8 = (mask.astype(np.uint8) * 255)

    if invert:
        mask_u8 = 255 - mask_u8

    img = Image.fromarray(mask_u8)
    img.save(out_path)

def save_boxes_overlay2(
    gray_img: np.ndarray,
    boxes: List["BoundingBox"],
    out_path: Path,
    membrane_bounds: Optional[Tuple[int, int]] = None,
    box_color=(0, 255, 0, 70),         # (R,G,B,A)
    boundary_color=(0, 200, 0, 180),   # (R,G,B,A)
    line_width: int = 2,
    boundary_width: int = 3,
) -> None:
    """
    输入必须是 numpy 灰度图：
      - float / int 均可
      - shape: (H, W)

    会先按你的规则做“反色可视化”：
      out = clip(255 - gray_img, 0, 255).astype(uint8)

    然后在反色后的图上叠加半透明缺陷框与膜边界，最后保存到 out_path。
    """
    if not isinstance(gray_img, np.ndarray):
        raise TypeError(f"gray_img must be np.ndarray, got {type(gray_img)}")
    if gray_img.ndim != 2:
        raise ValueError(f"gray_img must be 2D grayscale (H,W), got shape={gray_img.shape}")

    # ---------- 1) 按你的 _save_float_array_as_gray 方式反色并转 uint8 ----------
    arr = np.asarray(gray_img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    inv = 255.0 - arr
    inv = np.clip(inv, 0, 255).astype(np.uint8)

    # ---------- 2) 转 PIL，并准备 RGBA 叠加层 ----------
    base = Image.fromarray(inv).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    W, H = base.size

    # ---------- 3) 画膜边界 ----------
    if membrane_bounds is not None:
        left, right = membrane_bounds
        left = int(max(0, min(left, W - 1)))
        right = int(max(0, min(right, W - 1)))
        draw.line([(left, 0), (left, H - 1)], fill=boundary_color, width=boundary_width)
        draw.line([(right, 0), (right, H - 1)], fill=boundary_color, width=boundary_width)

    # ---------- 4) 画缺陷框（半透明边框；如需填充见注释） ----------
    for b in boxes:
        left = int(getattr(b, "left"))
        top = int(getattr(b, "top"))
        right = int(getattr(b, "right"))
        bottom = int(getattr(b, "bottom"))

        # 越界保护 + 方向修正
        left = max(0, min(left, W - 1))
        right = max(0, min(right, W - 1))
        top = max(0, min(top, H - 1))
        bottom = max(0, min(bottom, H - 1))
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top

        # 你之前的 +1 逻辑：画到像素边界更贴合
        r2 = min(W, right + 1)
        b2 = min(H, bottom + 1)

        draw.rectangle([(left, top), (r2, b2)], outline=box_color, width=line_width)

        # 若你要半透明填充，打开下面两行即可：
        # fill_color = (box_color[0], box_color[1], box_color[2], min(box_color[3], 60))
        # draw.rectangle([(left, top), (r2, b2)], fill=fill_color)

    # ---------- 5) 合成并保存 ----------
    out = Image.alpha_composite(base, overlay)

    ensure_directory(out_path.parent)

    # 统一转 RGB 保存，避免 jpg 不支持 alpha 的坑
    out.convert("RGB").save(out_path)

def save_boxes_overlay(
    gray_img: Image.Image,
    boxes: List[BoundingBox],
    out_path: Path,
    membrane_bounds: Optional[Tuple[int, int]] = None,
    box_color=(0, 255, 0, 70),   # 半透明绿色 (R,G,B,A)
    boundary_color=(0, 200, 0, 180)  # 膜边界更亮一点
):
    """
    在灰度图上叠加透明度可控的缺陷框与膜边界。
    """
    base = gray_img.convert("RGBA")   # 原图改为 RGBA 以支持透明叠加
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))  # 全透明图层
    draw = ImageDraw.Draw(overlay)

    h = base.height

    # ----------- 画膜边界（半透明绿线）-----------
    if membrane_bounds is not None:
        left, right = membrane_bounds

        draw.line([(left, 0), (left, h)], fill=boundary_color, width=3)
        draw.line([(right, 0), (right, h)], fill=boundary_color, width=3)

    # ----------- 画缺陷框（半透明）-----------
    for box in boxes:
        left, top, right, bottom = box.left, box.top, box.right, box.bottom
        
        # 半透明边框
        draw.rectangle(
            [(left, top), (right + 1, bottom + 1)],
            outline=box_color,
            width=2,
        )

        # 半透明填充（可选，若你不想填充就注释掉）
        # fill_color = (0, 255, 0, 60)  # 更淡的透明绿
        # draw.rectangle(
        #     [(left, top), (right + 1, bottom + 1)],
        #     fill=fill_color,
        # )

    # ----------- 合成透明层与原图 -----------
    out = Image.alpha_composite(base, overlay)

    ensure_directory(out_path.parent)
    out.convert("RGB").save(out_path)


def save_mask_overlay_red(
    gray_img: Image.Image,
    mask: np.ndarray,
    out_path: Path,
) -> None:
    """
    在灰度图上用红色标记 mask=True 的像素（你要求的“红点，而不是黑白 mask”）。
    """
    base = gray_img.convert("RGB")
    arr = np.array(base, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    arr[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)
    ensure_directory(out_path.parent)
    Image.fromarray(arr).save(out_path)


# ========= 分割核心函数 + 调试版本 =========

def _difference_map_core(
    region: np.ndarray,
    blur_mode: str,
    debug_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Multi-scale 版本：
      - 三个 box blur：size = 17, 61, 171
      - diff_x = blur_x - region
      - combined = max(diff17, diff61, diff171, 0)
      - 可选：debug_dir 保存所有中间结果
    """
    global _FALLBACK_WARNED

    # 三尺度卷积核
    sizes = (17, 61, 171)

    # ==== 1. 生成模糊图 (blur17 / blur61 / blur171) ====
    if blur_mode == "opencv":
        try:
            blur17 = box_blur_opencv(region, size=sizes[0])
            blur61 = box_blur_opencv(region, size=sizes[1])
            blur171 = box_blur_opencv(region, size=sizes[2])
        except RuntimeError:
            if not _FALLBACK_WARNED:
                print("[WARN] OpenCV blur unavailable; falling back to integral_shared.")
                _FALLBACK_WARNED = True
            return _difference_map_core(region, "integral_shared", debug_dir=debug_dir)

    elif blur_mode == "integral_shared":
        # 一次积分图共享计算
        blur_map = box_blur_pair_shared(region, sizes)
        blur17 = blur_map[sizes[0]]
        blur61 = blur_map[sizes[1]]
        blur171 = blur_map[sizes[2]]

    else:  # "integral"
        blur17 = box_blur(region, size=sizes[0])
        blur61 = box_blur(region, size=sizes[1])
        blur171 = box_blur(region, size=sizes[2])

    # ===== 2. 差分 =====
    diff17 = blur17 - region
    diff61 = blur61 - region
    diff171 = blur171 - region

    # 组合差分
    combined = np.maximum.reduce([diff17, diff61, diff171, np.zeros_like(region)])

    # ===== 3. Debug 输出 & 保存 =====
    if debug_dir is not None:
        ensure_directory(debug_dir)

        print("    ▶ [_difference_map_core] 三尺度差分函数")
        print(f"       - 输入 region 形状 = {region.shape}")
        print(f"       - blur17 : min={blur17.min():.2f}, max={blur17.max():.2f}")
        print(f"       - blur61 : min={blur61.min():.2f}, max={blur61.max():.2f}")
        print(f"       - blur171: min={blur171.min():.2f}, max={blur171.max():.2f}")
        print(f"       - diff17 : min={diff17.min():.2f}, max={diff17.max():.2f}")
        print(f"       - diff61 : min={diff61.min():.2f}, max={diff61.max():.2f}")
        print(f"       - diff171: min={diff171.min():.2f}, max={diff171.max():.2f}")
        print(f"       - combined: min={combined.min():.2f}, max={combined.max():.2f}")

        # 保存图像
        save_float_as_gray_raw(region, debug_dir / "roi_gray.png")
        save_float_as_gray_raw(blur17, debug_dir / "blur_17.png")
        save_float_as_gray_raw(blur61, debug_dir / "blur_61.png")
        save_float_as_gray_raw(blur171, debug_dir / "blur_171.png")

        _save_float_array_as_gray(diff17, debug_dir / "diff_17.png")
        _save_float_array_as_gray(diff61, debug_dir / "diff_61.png")
        _save_float_array_as_gray(diff171, debug_dir / "diff_171.png")
        _save_float_array_as_gray(combined, debug_dir / "combined_max_3scale.png")

    return combined



def compute_difference_map(
    gray_float: np.ndarray,
    left: int,
    right: int,
    blur_mode: str,
    debug_dir: Optional[Path] = None,
) -> np.ndarray:
    left = max(0, left)
    right = min(gray_float.shape[1] - 1, right)
    if right <= left:
        raise ValueError("ROI bounds are invalid for difference computation.")

    roi = gray_float[:, left : right + 1]
    print(f"  ▶ [Step 2 - 差分图] compute_difference_map: ROI=[:, {left}:{right + 1}], shape={roi.shape}")

    core_dir = debug_dir / "difference_core" if debug_dir is not None else None
    diff_roi = _difference_map_core(roi, blur_mode=blur_mode, debug_dir=core_dir)

    # 1) 只在 ROI 上计算行中位数
    row_median = np.median(diff_roi, axis=1, keepdims=True)
    print(
        f"      行方向中位数 row_median: "
        f"min={row_median.min():.6f}, "
        f"max={row_median.max():.6f}, "
        f"mean={row_median.mean():.6f}"
    )

    # 2) 先把 diff_roi 填回整幅图
    diff_full = np.zeros_like(gray_float, dtype=np.float32)
    diff_full[:, left : right + 1] = diff_roi

    # 3) 对整幅图做“行去中位数”（膜外本来就是 0，减去正数后会被 clip 成 0）
    difference_map_row = np.clip(diff_full - row_median, 0, None)
    print(
        f"      去行中位数后 difference_map_row: "
        f"min={difference_map_row.min():.3f}, max={difference_map_row.max():.3f}, "
        f"mean={difference_map_row.mean():.3f}"
    )

    # 4) 再做列中位数去趋势
    col_median = np.median(difference_map_row, axis=0, keepdims=True)
    print(
        f"      列方向中位数 col_median: "
        f"min={col_median.min():.6f}, "
        f"max={col_median.max():.6f}, "
        f"mean={col_median.mean():.6f}"
    )

    difference_map_row_col = np.clip(difference_map_row - col_median, 0, None)
    print(
        f"      去列中位数后 difference_map: "
        f"min={difference_map_row_col.min():.3f}, max={difference_map_row_col.max():.3f}, "
        f"mean={difference_map_row_col.mean():.3f}"
    )

    if debug_dir is not None:
        _save_float_array_as_gray(difference_map_row, debug_dir / "difference_map_row.png")
        _save_float_array_as_gray(difference_map_row_col, debug_dir / "difference_map_row_col.png")

    return difference_map_row_col


def determine_defect_threshold(
    values: np.ndarray,
    membrane_mean: Optional[float] = None,
    debug_stats: Optional[Dict[str, float]] = None,
) -> float:
    """
    与原算法一致，但额外记录更多统计信息：
      - 新增 p85, p87, p90, p93
      - 新增 black_threshold = mean + 2*std
    """
    flat = values.ravel().astype(np.float32)
    if flat.size == 0:
        if debug_stats is not None:
            debug_stats.update(
                {
                    "n": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "mad": 0.0,
                    "p85": 0.0,
                    "p87": 0.0,
                    "p90": 0.0,
                    "p93": 0.0,
                    "p95": 0.0,
                    "p96": 0.0,
                    "p97": 0.0,
                    "p98": 0.0,
                    "p99": 0.0,
                    "black_threshold1": 0.0,
                    "black_threshold2": 0.0,
                    "robust": 0.0,
                    "base": 0.0,
                    "scale": 1.0,
                    "threshold": 0.0,
                    "membrane_mean": float(membrane_mean) if membrane_mean is not None else -1.0,
                }
            )
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

    # === 核心统计量 ===
    mean_value = float(sample.mean())
    std_value = float(sample.std())
    median_value = float(np.median(sample))
    mad_value = float(np.median(np.abs(sample - median_value)))

    # === 新增分位数 ===
    percentile_85 = float(np.percentile(sample, 85))
    percentile_87 = float(np.percentile(sample, 87))
    percentile_90 = float(np.percentile(sample, 90))
    percentile_93 = float(np.percentile(sample, 93))

    # === 已有分位数 ===
    percentile_95 = float(np.percentile(sample, 95))
    percentile_96 = float(np.percentile(sample, 96))
    percentile_97 = float(np.percentile(sample, 97))
    percentile_98 = float(np.percentile(sample, 98))
    percentile_99 = float(np.percentile(sample, 99))

    # === mean + 2*std（新增 black_threshold）===
    
    # === base 仍用原逻辑 ===
    from config.constant import zsz_Constants
    robust0 =  median_value + 1.0 * (1.4826 * mad_value if mad_value > 0 else 0.0)

    base = max(zsz_Constants.MIN_GRAY, min(robust0, mean_value , zsz_Constants.MAX_GRAY))

    scale = 1.0
    threshold = base * scale

    black_threshold1 = mean_value + 2.0 * std_value
    black_threshold2 = median_value + 6.0 * (1.4826 * mad_value if mad_value > 0 else 0.0)
    robust = median_value + 3.0 * (1.4826 * mad_value if mad_value > 0 else 0.0)
    
    dark_margin = min(zsz_Constants.MAX_DARK, max(zsz_Constants.MIN_DARK,black_threshold1,black_threshold2,robust))
    # === 写入 debug_stats ===
    if debug_stats is not None:
        debug_stats.update(
            {
                "n": int(sample.size),
                "mean": mean_value,
                "std": std_value,
                "median": median_value,
                "mad": mad_value,

                "p85": percentile_85,
                "p87": percentile_87,
                "p90": percentile_90,
                "p93": percentile_93,

                "p95": percentile_95,
                "p96": percentile_96,
                "p97": percentile_97,
                "p98": percentile_98,
                "p99": percentile_99,

                "black_threshold1": black_threshold1,
                "black_threshold2": black_threshold2,
                "dark_margin": dark_margin,
                "robust": robust,
                "base": base,
                "scale": scale,
                "threshold": threshold,
                "membrane_mean": float(membrane_mean) if membrane_mean is not None else -1.0,
            }
        )

    return base,dark_margin



# ========= 整体一步步跑分割阶段 =========

def run_segmentation_debug(
    gray_array: np.ndarray,
    min_component: int,
    blur_mode: str,
    img_debug_dir: Path,
) -> Tuple[
    float,               # background_mean
    float,               # threshold_value
    List[BoundingBox],   # boxes
    np.ndarray,          # membrane_mask
    Tuple[int, int],     # membrane_bounds
    np.ndarray,          # difference_map (去列趋势后)
    np.ndarray,          # defect_mask_raw
    np.ndarray,          # defect_mask_after_dense
    Dict[str, float],    # threshold_stats
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
    print("  ===== 开始分割调试：run_segmentation_debug =====")
    h, w = gray_array.shape
    gray_float = gray_array.astype(np.float32)
    from detect_core.zsz.membrane_grad_core import build_membrane_mask_grad

    membrane_mask, membrane_bounds, background_mean = build_membrane_mask_grad(
        gray_float,
        smoothing_ratio=0.003,
        background_percentile=0.99,
        min_background_value=210,
    )

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

    # 2) difference_map
    diff_dir = img_debug_dir / "difference"
    difference_map = compute_difference_map(
        gray_float,
        membrane_bounds[0],
        membrane_bounds[1],
        blur_mode=blur_mode,
        debug_dir=diff_dir,
    )
    # 新增：把所有 < 15 的值强制设为 0
    from config  import constant
    difference_map = np.where(difference_map < constant.zsz_Constants.NOICE_GRAY, 0, difference_map)

    # 3) 阈值（只看膜内像素，此处即全图）
    print("  ▶ [Step 3 - 阈值计算] 计算缺陷阈值")
    membrane_values = difference_map[membrane_mask]
    thr_stats: Dict[str, float] = {}
    membrane_mean = float(membrane_values.mean()) if membrane_values.size > 0 else None
    if membrane_mean is not None:
        print(f"      膜内 difference_map 均值 membrane_mean = {membrane_mean:.3f}")
    else:
        print("      膜内没有有效像素，membrane_mean = None")

    threshold_value, dark_margin = determine_defect_threshold(
        membrane_values,
        membrane_mean=membrane_mean,
        debug_stats=thr_stats,
    )


    thr_stats["threshold_after_bg_adjust"] = threshold_value
    thr_txt_path = img_debug_dir / "threshold_stats.txt"
    ensure_directory(thr_txt_path.parent)
    with thr_txt_path.open("w", encoding="utf-8") as f:
        for k, v in thr_stats.items():
            f.write(f"{k}: {v}\n")

    print("      阈值候选统计（写入 threshold_stats.txt）：")
    for k, v in thr_stats.items():
        print(f"        - {k}: {v}")

    # 4) defect_mask（原始） = difference_map > threshold 且在膜内
    print("  ▶ [Step 4 - 原始缺陷 mask] 根据最终阈值生成 defect_mask_raw")
    defect_mask_raw = (difference_map > threshold_value) & membrane_mask
    n_raw = int(defect_mask_raw.sum())
    print(
        f"      defect_mask_raw 中 True 像素数={n_raw} "
        f"(占整图比例={n_raw / (h * w):.6f})"
    )

    # _save_bool_mask_as_gray(
    #     defect_mask_raw,
    #     img_debug_dir / "defect_mask_raw_gray.png",
    #     invert = True,
    # )

    # 5) suppress_dense_clusters
    print("  ▶ [Step 5 - suppress_dense_clusters] 对稠密区域做抑制")
    defect_mask_after = suppress_dense_clusters(
        defect_mask_raw,
        tile_size=256,
        density_threshold=0.08,
    )
    
    n_after = int(defect_mask_after.sum())
    print(
        f"      抑制后 defect_mask_after 中 True 像素数={n_after} "
        f"(占整图比例={n_after / (h * w):.6f})"
    )

    # _save_bool_mask_as_gray(
    #     defect_mask_after,
    #     img_debug_dir / "defect_mask_after_dense_gray.png",
    #     invert = True,
    # )

    # 6) extract_bounding_boxes（注意 dark_margin 的计算）
    # print("  ▶ [Step 6 - 连通域提取] 调用 extract_bounding_boxes")
    # dark_margin = DARK_RATIO_MARGIN_MIN
    # if threshold_value is not None:
    #     dark_margin = max(
    #         DARK_RATIO_MARGIN_MIN,
    #         threshold_value * DARK_RATIO_MARGIN_RATIO,
    #     )
    
    boxes = extract_bounding_boxes(
        defect_mask_after,
        min_pixels=min_component,
        reference_map=difference_map,
        reference_threshold=threshold_value,
        reference_margin=dark_margin,
    )
    ## 新增合并功能
    print(
        f"      extract_bounding_boxes 得到 boxes={len(boxes)} 个，"
        f"dark_margin={dark_margin:.3f}, threshold_final={threshold_value:.3f}"
    )
    from detect_core.zsz.box_merge import merge_overlapping_boxes
    boxes = merge_overlapping_boxes(boxes)

    print(
        f"      !!合并后boxes={len(boxes)} 个!!"
    )

    if boxes:
        sizes = [getattr(b, "pixels", 0) for b in boxes]
        print(
            f"        各连通域像素数统计: "
            f"min={min(sizes)}, max={max(sizes)}, "
            f"mean={sum(sizes)/len(sizes):.1f}"
        )

    print("  ===== 本图分割调试结束 =====")

    return (
        background_mean,
        threshold_value,
        boxes,
        membrane_mask,
        membrane_bounds,
        difference_map,
        defect_mask_raw,
        defect_mask_after,
        thr_stats,
    )


# ========= CLI / 主流程 =========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug segmentation (difference_map / threshold / suppress_dense_clusters / extract_bounding_boxes)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data"),
        help="输入图片文件夹（递归遍历）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./debug_seg_outputs"),
        help="输出结果根目录。",
    )
    parser.add_argument(
        "--min-component",
        type=int,
        default=8,
        help="extract_bounding_boxes 的最小连通域像素数。",
    )
    parser.add_argument(
        "--blur-mode",
        choices=BLUR_MODES,
        default="integral",
        help="difference_map 使用的 blur backend。",
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
        print(f"[WARN] 未在 {input_dir} 下找到图片。")
        return

    print(f"🔍 共发现 {len(image_files)} 张图片，将逐一进行分割调试。")
    print(f"    输入目录: {input_dir}")
    print(f"    输出目录: {output_dir}")
    print(f"    blur_mode: {blur_mode}, min_component: {min_component}")

    for idx, image_path in enumerate(image_files, 1):
        print("\n====================================================")
        print(f"📸 [{idx}/{len(image_files)}] 开始处理图像: {image_path.name}")
        img_debug_dir = output_dir / image_path.stem
        ensure_directory(img_debug_dir)

        print("  ▶ [准备] 加载灰度图")
        gray_img = load_grayscale(image_path)
        gray_arr = np.array(gray_img, dtype=np.float32)
        print(f"      灰度图 shape={gray_arr.shape}, dtype={gray_arr.dtype}")

        (
            background_mean,
            threshold_value,
            boxes,
            membrane_mask,
            membrane_bounds,
            difference_map,
            defect_mask_raw,
            defect_mask_after,
            thr_stats,
        ) = run_segmentation_debug(
            gray_arr,
            min_component=min_component,
            blur_mode=blur_mode,
            img_debug_dir=img_debug_dir,
        )

        print("  ▶ [保存可视化] 缺陷红点 overlay & 连通域框图")
        # save_mask_overlay_red(
        #     gray_img,
        #     defect_mask_raw,
        #     img_debug_dir / "defect_mask_raw_red_overlay.png",
        # )
        # save_mask_overlay_red(
        #     difference_map,
        #     defect_mask_after,
        #     img_debug_dir / "fff_defect_mask_after_dense_red_overlay.png",
        # )

        save_boxes_overlay2(
            difference_map,
            boxes,
            img_debug_dir / "difference_boxes_overlay.png",
            membrane_bounds=membrane_bounds,
        )
        save_boxes_overlay(
            gray_img,
            boxes,
            img_debug_dir / "original_boxes_overlay.png",
            membrane_bounds=membrane_bounds,
        )

        print(
            f"  [summary] 图像 {image_path.name}: "
            f"background={background_mean:.2f}, "
            f"threshold_final={threshold_value:.3f}, boxes={len(boxes)}"
        )

        gray_img.close()

    print("\n🎉 所有图片分割调试完成。输出已写入目录:", output_dir)


if __name__ == "__main__":
    main()
