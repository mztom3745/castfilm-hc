from __future__ import annotations

from typing import Tuple

import numpy as np


def _ensure_odd(value: int) -> int:
    if value % 2 == 0:
        value += 1
    return max(3, value)


def _smooth_signal(signal: np.ndarray, window_ratio: float) -> np.ndarray:
    window = max(5, int(len(signal) * window_ratio))
    window = _ensure_odd(window)
    pad = window//2
    sig_pad = np.pad(signal, pad, mode='edge')
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(sig_pad.astype(np.float32), kernel, mode="same")


def _fallback_bounds(width: int, ratio: float) -> Tuple[int, int]:
    margin = max(1, int(width * ratio))
    if margin * 2 >= width:
        margin = max(1, width // 4)
    start = margin
    end = max(start + 1, width - margin)
    return start, end


def _find_boundary_from_left(signal: np.ndarray, bg_level: float, start: int, threshold: float, min_run: int) -> int:
    run = 0
    candidate = start
    for idx in range(start, signal.size):
        if bg_level - signal[idx] >= threshold:
            run += 1
            if run >= min_run:
                candidate = idx - run + 1
                break
        else:
            run = 0
    return candidate


def _find_boundary_from_right(signal: np.ndarray, bg_level: float, start: int, threshold: float, min_run: int) -> int:
    run = 0
    candidate = start
    for idx in range(start, -1, -1):
        if bg_level - signal[idx] >= threshold:
            run += 1
            if run >= min_run:
                candidate = idx + run - 1
                break
        else:
            run = 0
    return candidate

# ratio
# → 左右用于估计背景的带宽比例；越大，背景估计越稳，但越容易吃到膜的一部分（尤其膜很宽时）
# smoothing_ratio
# → 列均值平滑窗口占全宽的比例；越大，趋势越平滑，但边界会变钝，不宜太大
# background_percentile
# → 取背景带中较亮的一部分作为“背景亮度”；从 0.9→0.99 会更偏向最亮像素，对局部暗点更鲁棒
# drop_ratio
# → 把“比背景亮度低多少”视为膜；比例越大，只有更明显变暗才会被认定为膜 → 边界更靠内
# min_run_ratio
# → 要求连续多少列变暗才认为是真边界；越大越不容易被局部噪声影响
# edge_clip_ratio
# → 对检测出的原始边界进行内缩；用于避免把边缘不稳定区域算进膜中
def locate_membrane_bounds(
    gray: np.ndarray,
    ratio: float = 0.1,
    smoothing_ratio: float = 0.003,
    background_percentile: float = 0.95,
    drop_ratio: float = 0.03,
    min_run_ratio: float = 0.002,
) -> Tuple[int, int]:
    """Locate membrane bounds by comparing smoothed column averages to background."""
    if gray.ndim != 2:
        raise ValueError("Membrane detection expects a 2D grayscale array.")
    _, width = gray.shape
    if width == 0:
        raise ValueError("Image width must be positive.")
    column_means = gray.mean(axis=0)
    #每一列的平均灰度 → 长度 = width 的一维信号
    smoothed = _smooth_signal(column_means, smoothing_ratio)
    ##smooth_signal 做一维滑动平均平滑
    bg_width = max(5, int(width * ratio))
    left_band = smoothed[:bg_width]
    right_band = smoothed[-bg_width:]
    percentile = np.clip(background_percentile * 100.0, 0.0, 100.0)
    left_bg = float(np.percentile(left_band, percentile))
    right_bg = float(np.percentile(right_band, percentile))
    left_threshold = max(5.0, drop_ratio * abs(left_bg))
    ##把“背景亮度下降多少算膜”转成阈值
    right_threshold = max(5.0, drop_ratio * abs(right_bg))
    min_run = max(3, int(width * min_run_ratio))
    left_start = bg_width
    right_start = width - bg_width - 1
    left = _find_boundary_from_left(smoothed, left_bg, left_start, left_threshold, min_run)
    ##从左向右扫描列均值 smoothed[x]
    ##找一段连续 min_run 列，都满足：
    ##背景亮度 - 当前列亮度 >= 某个下降阈值
    ##第一次满足条件时，把这一段的起点 idx - run + 1当作左膜边界
    right = _find_boundary_from_right(smoothed, right_bg, right_start, right_threshold, min_run)
    if right <= left:
        return _fallback_bounds(width, ratio)
    # 如果检测出来的 left/right 不合法（例如 right <= left），就采用简单的对称边界：
    # 左边界 = width * ratio
    # 右边界 = width - ratio * width
    # 也就是“假定膜占中间 1 - 2*ratio 的区域”，默认 ratio=0.1 → 中间 80%
    return left, right


def build_membrane_mask(
    gray: np.ndarray,
    ratio: float = 0.1,
    smoothing_ratio: float = 0.003,
    background_percentile: float = 0.95,
    drop_ratio: float = 0.03,
    min_run_ratio: float = 0.002,
    edge_clip_ratio: float = 0.05,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return the membrane mask together with (left, right) bounds."""
    raw_left, raw_right = locate_membrane_bounds(
        gray,
        ratio=ratio,
        smoothing_ratio=smoothing_ratio,
        background_percentile=background_percentile,
        drop_ratio=drop_ratio,
        min_run_ratio=min_run_ratio,
    )
    if raw_right <= raw_left:
        raw_left, raw_right = _fallback_bounds(gray.shape[1], ratio)
    clip = max(1, int((raw_right - raw_left + 1) * edge_clip_ratio))
    left = min(max(0, raw_left + clip), gray.shape[1] - 2)
    right = max(min(gray.shape[1] - 1, raw_right - clip), left + 1)
    mask = np.zeros_like(gray, dtype=bool)
    mask[:, left : right + 1] = True

    return mask, (left, right)
