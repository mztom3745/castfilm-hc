from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # 用于画曲线

from detect_core.castfilm_detection.processing.membrane import (
    build_membrane_mask,  # 原始实现，用来对比
)
from detect_core.castfilm_detection.common.filesystem import (
    ensure_directory,
    iter_image_files,
)

############################################
# 1. 工具函数：和 membrane.py 中保持相同/兼容逻辑
############################################


def _ensure_odd(value: int) -> int:
    if value % 2 == 0:
        value += 1
    return max(3, value)


def _smooth_signal(signal: np.ndarray, window_ratio: float) -> np.ndarray:
    """
    与库中一致：窗口大小 = max(5, len * window_ratio)，并保证为奇数。
    """
    window = max(5, int(len(signal) * window_ratio))
    window = _ensure_odd(window)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _fallback_bounds(width: int, ratio: float) -> Tuple[int, int]:
    margin = max(1, int(width * ratio))
    if margin * 2 >= width:
        margin = max(1, width // 4)
    start = margin
    end = max(start + 1, width - margin)
    return start, end


#########################################################
# 1.a 中心扫描背景 + 膜边界定位（新算法的核心）
#########################################################


def _locate_membrane_from_center(
    smoothed: np.ndarray,
    bg_window: int = 200,
    bg_tol: float = 3.0,
    bg_min_level: float = 210.0,
    inner_offset: int = 50,
    max_half_width: int = 7000,
    default_bg: float = 220.0,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    从信号中心向左右扫描背景区域，并推断膜左右边界。

    背景区域定义：
        - 在长度 bg_window 的窗口内：max - min <= bg_tol
        - 且窗口均值 >= bg_min_level

    策略：
        - 从中心向左扫描：找到第一个满足条件的窗口 [start_l, end_l)
        - 从中心向右扫描：找到第一个满足条件的窗口 [start_r, end_r)
        - 左边：背景窗口靠近中心的一端是 end_l-1，从这里向“中心方向”再走 inner_offset 个像素，
                作为膜左边界 raw_left。
        - 右边：背景窗口靠近中心的一端是 start_r，从这里向“中心方向”反向走 inner_offset 个像素，
                作为膜右边界 raw_right。
        - 若某侧没有背景，则该侧用中心±max_half_width 像素兜底；
        - 若两侧都没有背景，背景均值 bg_mean=default_bg，边界仍然按兜底规则。
    """
    width = smoothed.size
    center = width // 2
    max_half_width = min(max_half_width, center)  # 不越界

    print(
        f"➡️ 中心扫描背景: width={width}, center={center}, "
        f"bg_window={bg_window}, bg_tol={bg_tol}, bg_min_level={bg_min_level}, "
        f"inner_offset={inner_offset}, max_half_width={max_half_width}"
    )

    def scan_left() -> Tuple[int | None, int | None, float | None]:
        """从中心向左寻找背景窗口，返回 (start, end, mean)。"""
        start_min = max(0, center - max_half_width - bg_window)
        start_max = max(0, center - bg_window)
        found_start = None
        found_end = None
        found_mean = None

        for start in range(start_max, start_min - 1, -1):
            end = start + bg_window
            if end > center:  # 要求窗口整体在中心左侧
                continue
            window = smoothed[start:end]
            w_min = float(window.min())
            w_max = float(window.max())
            w_mean = float(window.mean())
            if (w_max - w_min) <= bg_tol and w_mean >= bg_min_level:
                found_start, found_end, found_mean = start, end, w_mean
                print(
                    f"   ✔ 左侧背景窗口: [{found_start}, {found_end}) "
                    f"min={w_min:.3f}, max={w_max:.3f}, mean={w_mean:.3f}"
                )
                break

        if found_start is None:
            print("   ⚠ 左侧未找到满足条件的背景窗口")
        return found_start, found_end, found_mean

    def scan_right() -> Tuple[int | None, int | None, float | None]:
        """从中心向右寻找背景窗口，返回 (start, end, mean)。"""
        start_min = center
        start_max = min(width - bg_window, center + max_half_width)
        found_start = None
        found_end = None
        found_mean = None

        for start in range(start_min, start_max + 1):
            end = start + bg_window
            if start < center:  # 要求窗口整体在中心右侧
                continue
            if end > width:
                break
            window = smoothed[start:end]
            w_min = float(window.min())
            w_max = float(window.max())
            w_mean = float(window.mean())
            if (w_max - w_min) <= bg_tol and w_mean >= bg_min_level:
                found_start, found_end, found_mean = start, end, w_mean
                print(
                    f"   ✔ 右侧背景窗口: [{found_start}, {found_end}) "
                    f"min={w_min:.3f}, max={w_max:.3f}, mean={w_mean:.3f}"
                )
                break

        if found_start is None:
            print("   ⚠ 右侧未找到满足条件的背景窗口")
        return found_start, found_end, found_mean

    left_start, left_end, left_mean = scan_left()
    right_start, right_end, right_mean = scan_right()

    # 背景均值与来源
    bg_values = []
    if left_mean is not None:
        bg_values.append(left_mean)
    if right_mean is not None:
        bg_values.append(right_mean)

    if bg_values:
        bg_mean = float(np.mean(bg_values))
        if left_mean is not None and right_mean is not None:
            bg_source = "both_center_scan"
        elif left_mean is not None:
            bg_source = "left_only_center_scan"
        else:
            bg_source = "right_only_center_scan"
    else:
        bg_mean = float(default_bg)
        bg_source = "default_220"

    # 左右 raw 边界
    if left_end is not None:
        # 背景靠近中心的那一端是 end-1，向中心再走 inner_offset 个像素
        inner_edge = left_end - 1
        raw_left = min(center - 1, inner_edge + inner_offset)
        raw_left = max(0, raw_left)
    else:
        raw_left = max(0, center - max_half_width)
        print(
            f"   ⚠ 左界使用兜底: raw_left = center - {max_half_width} = {raw_left}"
        )

    if right_start is not None:
        # 背景靠近中心的那一端是 start，向中心再走 inner_offset 个像素
        inner_edge = right_start
        raw_right = max(center, inner_edge - inner_offset)
        raw_right = min(width - 1, raw_right)
    else:
        raw_right = min(width - 1, center + max_half_width)
        print(
            f"   ⚠ 右界使用兜底: raw_right = center + {max_half_width} = {raw_right}"
        )

    if raw_right <= raw_left:
        print(
            f"   ⚠ 中心扫描得到 raw_left={raw_left}, raw_right={raw_right} 非法，改用全图兜底"
        )
        raw_left, raw_right = 0, width - 1

    print(
        f"   ➡️ 中心扫描结果: bg_mean={bg_mean:.3f}, "
        f"raw_left={raw_left}, raw_right={raw_right}, bg_source={bg_source}"
    )

    scalars_center: Dict[str, Any] = {
        "bg_mean": bg_mean,
        "bg_source": bg_source,
        "center": center,
        "bg_window": bg_window,
        "bg_tol": bg_tol,
        "bg_min_level": bg_min_level,
        "inner_offset": inner_offset,
        "max_half_width": max_half_width,
        "left_bg": float(left_mean) if left_mean is not None else bg_mean,
        "right_bg": float(right_mean) if right_mean is not None else bg_mean,
        "left_bg_start": left_start if left_start is not None else -1,
        "left_bg_end": left_end if left_end is not None else -1,
        "right_bg_start": right_start if right_start is not None else -1,
        "right_bg_end": right_end if right_end is not None else -1,
        "left_band_len": int(bg_window) if left_start is not None else 0,
        "right_band_len": int(bg_window) if right_start is not None else 0,
    }

    return raw_left, raw_right, scalars_center


############################################
# 1.b debug_build_membrane_mask：使用中心扫描算法
############################################


def debug_build_membrane_mask(
    gray: np.ndarray,
    ratio: float = 0.1,
    smoothing_ratio: float = 0.003,
    background_percentile: float = 0.95,  # 兼容参数，不再用于新算法
    drop_ratio: float = 0.03,             # 兼容参数，不再用于新算法
    min_run_ratio: float = 0.002,         # 兼容参数，不再用于新算法
    edge_clip_ratio: float = 0.05,
) -> Tuple[np.ndarray, Tuple[int, int], dict, dict]:
    """
    我们自己的 debug 版 build_membrane_mask（新算法版本）：
    - 不再依赖左右 ratio 区域，而是从“中心向两侧”寻找背景窗口；
    - 利用背景平稳 + 膜/背景之间的明显变化来推断原始边界 raw_left/raw_right；
    - 再按照 edge_clip_ratio 对边界做一点内缩，得到最终的 (left, right)；
    - 返回 mask + (left,right) + 调试信息。
    """
    if gray.ndim != 2:
        raise ValueError("Membrane detection expects a 2D grayscale array.")
    height, width = gray.shape
    if width == 0:
        raise ValueError("Image width must be positive.")

    # 1) 列均值 + 平滑
    column_means = gray.mean(axis=0)
    smoothed = _smooth_signal(column_means, smoothing_ratio)
    print(f"➡️ 已计算列均值与平滑信号: width={width}")

    # 2) 使用中心扫描算法定位 raw_left/raw_right
    raw_left, raw_right, center_scalars = _locate_membrane_from_center(
        smoothed,
        bg_window=max(100, int(width * 0.01)),  # 窗口长度按宽度自适应
        bg_tol=3.0,
        bg_min_level=210.0,
        inner_offset=50,
        max_half_width=7000,
        default_bg=220.0,
    )

    used_fallback = False
    if raw_right <= raw_left:
        # 极限兜底（理论上上面已经处理过）
        used_fallback = True
        fb_left, fb_right = _fallback_bounds(width, ratio)
        print(
            f"⚠ 中心扫描后 raw_right({raw_right}) <= raw_left({raw_left})，"
            f"再次启用 fallback: fb_left={fb_left}, fb_right={fb_right}"
        )
        raw_left, raw_right = fb_left, fb_right

    # 3) edge_clip 内缩
    clip = max(1, int((raw_right - raw_left + 1) * edge_clip_ratio))
    left = min(max(0, raw_left + clip), width - 2)
    right = max(min(width - 1, raw_right - clip), left + 1)

    # 4) 生成 mask
    mask = np.zeros_like(gray, dtype=bool)
    mask[:, left : right + 1] = True

    print(
        f"   ➡️ clip 像素: {clip}, final_left={left}, final_right={right}, "
        f"final_width={right-left+1} ({(right-left+1)/float(width)*100:.1f}% of width)"
    )

    # 5) scalars / arrays 返回调试信息
    scalars: Dict[str, Any] = {
        "height": height,
        "width": width,
        "ratio": ratio,
        "smoothing_ratio": smoothing_ratio,
        "background_percentile": background_percentile,
        "drop_ratio": drop_ratio,
        "min_run_ratio": min_run_ratio,
        "bg_width": center_scalars["bg_window"],
        "bg_source": center_scalars["bg_source"],
        "left_band_len": center_scalars["left_band_len"],
        "right_band_len": center_scalars["right_band_len"],
        "percentile": float(np.clip(background_percentile * 100.0, 0.0, 100.0)),
        "left_bg": float(center_scalars["left_bg"]),
        "right_bg": float(center_scalars["right_bg"]),
        "left_threshold": 0.0,   # 新算法未使用梯度阈值，保留字段方便打印
        "right_threshold": 0.0,
        "min_run": 0,
        "left_start": center_scalars["left_bg_start"],
        "right_start": center_scalars["right_bg_start"],
        "raw_left": raw_left,
        "raw_right": raw_right,
        "used_fallback": used_fallback,
        "edge_clip_ratio": edge_clip_ratio,
        "clip_pixels": clip,
        "final_left": left,
        "final_right": right,
        "final_width": right - left + 1,
        "final_width_ratio": (right - left + 1) / float(width),
        "bg_mean": center_scalars["bg_mean"],
        "center": center_scalars["center"],
    }

    arrays = {
        "column_means": column_means,
        "smoothed": smoothed,
        # 新算法的“背景带”就是找到的背景窗口，可用于调试
        "left_band": (
            smoothed[
                center_scalars["left_bg_start"] : center_scalars["left_bg_end"]
            ].astype(np.float32)
            if center_scalars["left_bg_start"] >= 0
            else np.array([], dtype=np.float32)
        ),
        "right_band": (
            smoothed[
                center_scalars["right_bg_start"] : center_scalars["right_bg_end"]
            ].astype(np.float32)
            if center_scalars["right_bg_start"] >= 0
            else np.array([], dtype=np.float32)
        ),
    }

    return mask, (left, right), scalars, arrays


############################################
# 2. I/O 辅助：读图 + 可视化膜区域
############################################


def load_grayscale(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    return img.convert("L")


def save_membrane_overlay(
    gray_img: Image.Image,
    membrane_mask: np.ndarray,
    out_path: Path,
) -> None:
    base = gray_img.convert("RGB")
    arr = np.array(base, dtype=np.uint8)
    mask = membrane_mask.astype(bool)

    highlight_color = np.array([0, 255, 0], dtype=np.uint8)
    arr[mask] = (
        0.5 * arr[mask].astype(np.float32) + 0.5 * highlight_color.astype(np.float32)
    ).astype(np.uint8)

    out_img = Image.fromarray(arr)
    ensure_directory(out_path.parent)
    out_img.save(out_path)


############################################
# 3. 单张图像的调试逻辑
############################################


def analyze_single_image(
    image_path: Path,
    out_dir: Path,
    ratio: float,
    smoothing_ratio: float,
    background_percentile: float,
    drop_ratio: float,
    min_run_ratio: float,
    edge_clip_ratio: float,
) -> None:
    ensure_directory(out_dir)

    gray_img = load_grayscale(image_path)
    gray_array = np.array(gray_img, dtype=np.float32)

    print(f"\n📸 分析图像: {image_path.name}")
    print(f"   shape = {gray_array.shape}")

    # 1) 我们自己的 debug 版（中心扫描算法）
    dbg_mask, dbg_bounds, scalars, arrays = debug_build_membrane_mask(
        gray_array,
        ratio=ratio,
        smoothing_ratio=smoothing_ratio,
        background_percentile=background_percentile,
        drop_ratio=drop_ratio,
        min_run_ratio=min_run_ratio,
        edge_clip_ratio=edge_clip_ratio,
    )

    # 2) 调用真正库里的 build_membrane_mask 对比
    lib_mask, lib_bounds = build_membrane_mask(
        gray_array,
        ratio=ratio,
        smoothing_ratio=smoothing_ratio,
        background_percentile=background_percentile,
        drop_ratio=drop_ratio,
        min_run_ratio=min_run_ratio,
        edge_clip_ratio=edge_clip_ratio,
    )

    same_bounds = (dbg_bounds == lib_bounds)
    same_mask = np.array_equal(dbg_mask, lib_mask)

    print(
        f"   bg_source = {scalars['bg_source']}, "
        f"left_band_len = {scalars['left_band_len']}, "
        f"right_band_len = {scalars['right_band_len']}"
    )
    print(f"   bg_mean = {scalars['bg_mean']:.3f}")
    print(f"   raw_left = {scalars['raw_left']}, raw_right = {scalars['raw_right']}")
    print(f"   final_left = {scalars['final_left']}, final_right = {scalars['final_right']}")
    print(
        f"   lib_left = {lib_bounds[0]}, lib_right = {lib_bounds[1]} "
        f"(same_bounds={same_bounds})"
    )
    print(
        f"   used_fallback = {scalars['used_fallback']}, "
        f"edge_clip_ratio = {scalars['edge_clip_ratio']}, clip_pixels = {scalars['clip_pixels']}"
    )
    print(
        f"   final_width = {scalars['final_width']} "
        f"({scalars['final_width_ratio'] * 100:.1f}% of width)"
    )
    print(
        f"   与库中 build_membrane_mask 比较: "
        f"bounds_same={same_bounds}, mask_same={same_mask}"
    )

    # 3) 写出 debug 文本（报告）
    debug_txt = out_dir / "membrane_debug.txt"
    with debug_txt.open("w", encoding="utf-8") as f:
        f.write(f"image_name = {image_path.name}\n")
        for k, v in scalars.items():
            f.write(f"{k} = {v}\n")
        f.write(f"lib_left = {lib_bounds[0]}, lib_right = {lib_bounds[1]}\n")
        f.write(f"bounds_same_with_library = {same_bounds}\n")
        f.write(f"mask_same_with_library = {same_mask}\n")
        f.write("signal_plot = column_signals.png\n")

    # 4) 保存一维信号数组（txt）
    column_means = arrays["column_means"]
    smoothed = arrays["smoothed"]
    left_band = arrays["left_band"]
    right_band = arrays["right_band"]

    np.savetxt(out_dir / "column_means.txt", column_means, fmt="%.6f")
    np.savetxt(out_dir / "smoothed.txt", smoothed, fmt="%.6f")
    np.savetxt(out_dir / "left_band.txt", left_band, fmt="%.6f")
    np.savetxt(out_dir / "right_band.txt", right_band, fmt="%.6f")

    # 5) 画曲线：最后一张图中同时画出新算法 + 库算法的边界
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x_full = np.arange(column_means.size)
    width = scalars["width"]

    dbg_left, dbg_right = dbg_bounds
    lib_left, lib_right = lib_bounds

    # (0,0) 全局列均值 & 平滑
    ax0 = axes[0, 0]
    ax0.plot(x_full, column_means, label="column_means")
    ax0.plot(x_full, smoothed, label="smoothed")
    ax0.axvline(dbg_left, color="g", linestyle="--", label="ours_left")
    ax0.axvline(dbg_right, color="r", linestyle="--", label="ours_right")
    ax0.axvline(lib_left, color="c", linestyle=":", label="lib_left")
    ax0.axvline(lib_right, color="m", linestyle=":", label="lib_right")
    ax0.set_title("column_means & smoothed (full width)")
    ax0.set_xlabel("column index")
    ax0.set_ylabel("mean gray")
    ax0.legend(loc="best")

    # (0,1) 左背景窗口（如果有的话）
    ax1 = axes[0, 1]
    if left_band.size > 0:
        x_left = np.arange(left_band.size)
        ax1.plot(x_left, left_band, label="left_bg_window")
    ax1.set_title("left background window (center-scan)")
    ax1.set_xlabel("index within left_band")
    ax1.set_ylabel("value")
    ax1.legend(loc="best")

    # (1,0) 右背景窗口（如果有的话）
    ax2 = axes[1, 0]
    if right_band.size > 0:
        x_right = np.arange(right_band.size)
        ax2.plot(x_right, right_band, label="right_bg_window")
    ax2.set_title("right background window (center-scan)")
    ax2.set_xlabel("index within right_band")
    ax2.set_ylabel("value")
    ax2.legend(loc="best")

    # (1,1) ROI：同时画新算法和库算法的边界（最终对比）
    ax3 = axes[1, 1]
    pad = max(10, int(0.05 * width))
    left_view = max(0, min(dbg_left, lib_left) - pad)
    right_view = min(width, max(dbg_right, lib_right) + pad)
    x_roi = np.arange(left_view, right_view)
    ax3.plot(
        x_roi,
        column_means[left_view:right_view],
        label="column_means_roi",
    )
    ax3.plot(
        x_roi,
        smoothed[left_view:right_view],
        label="smoothed_roi",
    )
    ax3.axvline(dbg_left, color="g", linestyle="--", label="ours_left")
    ax3.axvline(dbg_right, color="r", linestyle="--", label="ours_right")
    ax3.axvline(lib_left, color="c", linestyle=":", label="lib_left")
    ax3.axvline(lib_right, color="m", linestyle=":", label="lib_right")
    ax3.set_title("ROI around membrane bounds (ours vs lib)")
    ax3.set_xlabel("column index")
    ax3.set_ylabel("mean gray")
    ax3.legend(loc="best")

    plt.tight_layout()
    plot_path = out_dir / "column_signals.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # 6) 保存膜 overlay 图（用库版看实际效果）
    overlay_path = out_dir / "membrane_overlay.png"
    save_membrane_overlay(gray_img, lib_mask, overlay_path)

    gray_img.close()


############################################
# 4. 命令行入口
############################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug build_membrane_mask correctness on a folder of images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data"),
        help="输入图片文件夹（会递归遍历所有图片）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./membrane_debug_outputs"),
        help="输出结果根目录。",
    )
    # 以下参数和原始 build_membrane_mask 保持一致，方便调参
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--smoothing-ratio", type=float, default=0.003)
    parser.add_argument("--background-percentile", type=float, default=0.95)
    parser.add_argument("--drop-ratio", type=float, default=0.03)
    parser.add_argument("--min-run-ratio", type=float, default=0.002)
    parser.add_argument("--edge-clip-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input
    output_root: Path = args.output

    image_files: List[Path] = list(iter_image_files(input_dir))
    if not image_files:
        print(f"[WARN] 未在目录 {input_dir} 下找到图片文件")
        return

    print(f"🔍 共找到 {len(image_files)} 张图片，将逐一分析 build_membrane_mask ...")

    for idx, image_path in enumerate(image_files, 1):
        img_out_dir = output_root / image_path.stem
        analyze_single_image(
            image_path=image_path,
            out_dir=img_out_dir,
            ratio=args.ratio,
            smoothing_ratio=args.smoothing_ratio,
            background_percentile=args.background_percentile,
            drop_ratio=args.drop_ratio,
            min_run_ratio=args.min_run_ratio,
            edge_clip_ratio=args.edge_clip_ratio,
        )
        print(f"   ✅ [{idx}/{len(image_files)}] 结果已写入 {img_out_dir}")

    print(
        "\n🎉 build_membrane_mask 调试完成，可逐张查看 membrane_debug.txt、"
        "column_signals.png 和中间数组。"
    )


if __name__ == "__main__":
    main()
