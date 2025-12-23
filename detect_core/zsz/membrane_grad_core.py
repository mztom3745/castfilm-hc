from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


def _ensure_odd(v: int) -> int:
    return v + 1 if v % 2 == 0 else v


def smooth_signal(signal: np.ndarray, window_ratio: float = 0.003) -> np.ndarray:
    """
    对 1D 信号进行平滑卷积，使用 edge-padding（复制信号两端值）避免 0 填充导致边缘失真。
    window = max(5, int(len * window_ratio))，并强制为奇数。
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = signal.size
    if n == 0:
        # 空信号，直接返回空拷贝
        return signal.copy()

    window = max(5, int(n * window_ratio))
    window = _ensure_odd(window)
    pad = window // 2

    padded = np.pad(signal, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _find_left_side(
    high_idx: np.ndarray,
    center: int,
    half_window: int,
    width: int,
    gray: np.ndarray,
    min_background_value: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    左侧逻辑（严格按照设定）：

    1）先假设中间的 10000 像素是膜：粗略膜区 [center-5000, center+5000]
    2）以左边为例：从 “center-5000 这一列” 往更左边（更外侧）找高梯度
       → 只考虑 <= left_guess 的高梯度索引
    3）遇到的第一个高梯度（离 left_guess 最近的那个）：
       → 向左移动 5 个像素，作为膜左边界
    4）继续向左看最外侧的高梯度
       → 这个点再往外的区域视为背景区域；对“最后一个高梯度的外侧区域”做均值检查
    """
    # 中间 10000 像素都是膜 ⇒ 左侧粗略膜边界
    left_guess = max(0, center - half_window)  # half_window=5000

    # 梯度索引范围 [0, width-2]，左侧“往外找” ⇒ 只看 <= left_guess 的高梯度
    search_start = 0
    search_end = min(left_guess, width - 2)

    side_idx = high_idx[(high_idx >= search_start) & (high_idx <= search_end)]
    side_idx_sorted = np.sort(side_idx)

    if side_idx_sorted.size == 0:
        # 搜索区间内没有任何高梯度点 → 无法确定左侧边界
        return None, None

    # 第一个高梯度：从粗略膜边界往外看，离 left_guess 最近的那个 → max(...)
    first_high = int(side_idx_sorted.max())

    # 最外侧的高梯度：越往 0 方向最小的索引
    last_high = int(side_idx_sorted.min())

    # 真正的膜左边界：在 first_high 基础上再向左偏移 5 像素
    left_boundary = max(0, first_high - 5)

    # 背景区域：最后一个高梯度值的外部区域
    # 这里按语义，将 [0, last_high] 左侧理解为背景（或者说“膜外背景区域在 last_high 更外一段”）
    # 为了简单起见，这里仍取 [0, last_high] 作为背景块做均值检查
    bg_end_col = last_high - 1
    if bg_end_col <= 0:
        # 没有可用的左侧背景区域
        return left_boundary, None

    bg_region = gray[:, 0 : bg_end_col + 1]
    bg_mean_val = float(bg_region.mean())

    if bg_mean_val < min_background_value:
        # 背景均值太低 → 左侧边界视为无效
        return None, None

    return left_boundary, bg_mean_val


def _find_right_side(
    high_idx: np.ndarray,
    center: int,
    half_window: int,
    width: int,
    gray: np.ndarray,
    min_background_value: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    右侧逻辑与左侧对称：

    1）先假设中间的 10000 像素是膜：粗略膜区 [center-5000, center+5000]
    2）以右边为例：从 “center+5000 这一列” 往更右边（更外侧）找高梯度
       → 只考虑 >= right_guess 的高梯度索引
    3）遇到的第一个高梯度（离 right_guess 最近的那个）：
       → 向右移动 5 个像素，作为膜右边界
    4）继续向右看最外侧的高梯度
       → 这个点再往外的区域视为背景区域；对“最后一个高梯度的外侧区域”做均值检查
    """
    max_grad_idx = max(0, width - 2)
    right_guess = min(max_grad_idx, center + half_window)

    # 右侧“往外找” ⇒ 只看 >= right_guess 的高梯度
    search_start = right_guess
    search_end = max_grad_idx

    side_idx = high_idx[(high_idx >= search_start) & (high_idx <= search_end)]
    side_idx_sorted = np.sort(side_idx)

    if side_idx_sorted.size == 0:
        # 搜索区间内没有任何高梯度点 → 无法确定右侧边界
        return None, None

    # 第一个高梯度：从粗略膜边界往外看，离 right_guess 最近的那个 → min(...)
    first_high = int(side_idx_sorted.min())

    # 最外侧的高梯度：越往右越大
    last_high = int(side_idx_sorted.max())

    # 真正的膜右边界：在 first_high 基础上往右偏移 5 像素
    right_boundary = min(width - 1, first_high + 5)

    # 背景区域：最后一个高梯度值的外部区域 → [last_high+1, width-1]
    bg_start_col = last_high + 1
    if bg_start_col >= width:
        # 没有可用的右侧背景区域
        return right_boundary, None

    bg_region = gray[:, bg_start_col:width]
    bg_mean_val = float(bg_region.mean())

    if bg_mean_val < min_background_value:
        # 背景均值太低 → 右侧边界视为无效
        return None, None

    return right_boundary, bg_mean_val

from config.constant import zsz_Constants

def build_membrane_mask_grad(
    gray: np.ndarray,
    ratio: float = 0.1,
    smoothing_ratio: float = 0.005,
    background_percentile: float = 0.995,
    drop_ratio: float = 0.03,
    min_run_ratio: float = 0.002,
    edge_clip_ratio: float = 0.05,
    min_background_value: float = zsz_Constants.MIN_BACKGROUND,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    基于梯度 + “中间 10000 像素视为膜” + “从中间向两侧 5000 像素往外找高梯度”的膜边界检测版本。
    """

    gray = np.asarray(gray, dtype=np.float32)
    h, width = gray.shape

    if width < 2:
        # 宽度太小，无法做梯度 → 直接整幅视为膜区域，背景值设为 220
        mask = np.ones_like(gray, dtype=bool)
        return mask, (0, width - 1), zsz_Constants.Normal_BACKGROUND

    # 1. 列均值 + 平滑
    column_means = gray.mean(axis=0)
    smoothed = smooth_signal(column_means, window_ratio=smoothing_ratio)

    # 2. 一阶梯度及绝对值（长度 = width - 1）
    grad = np.diff(smoothed)
    grad_abs = np.abs(grad)

    # 3. 用分位数选择高梯度阈值
    perc = float(np.clip(background_percentile * 100.0, 0.0, 100.0))
    thr = float(np.percentile(grad_abs, perc))
    high_idx = np.where(grad_abs >= thr)[0]  # index 范围 [0, width-2]

    # 4. “中间 10000 像素是膜” + 从中间向两侧 5000 像素往外找高梯度
    center = width // 2
    from config.constant import zsz_Constants

    half_window = zsz_Constants.BASE_HALF_PIXEL  # 即“最中间的 10000 像素都是膜，中间±5000”

    left_boundary, left_bg_mean = _find_left_side(
        high_idx=high_idx,
        center=center,
        half_window=half_window,
        width=width,
        gray=gray,
        min_background_value=min_background_value,
    )

    right_boundary, right_bg_mean = _find_right_side(
        high_idx=high_idx,
        center=center,
        half_window=half_window,
        width=width,
        gray=gray,
        min_background_value=min_background_value,
    )

    # 5. 异常处理：没有找到某一侧边界 → 使用 center ± half_window 回退
    if left_boundary is None:
        left_boundary = max(0, center - half_window)
    if right_boundary is None:
        right_boundary = min(width - 1, center + half_window)

    # 防止 left >= right 的极端情况
    if left_boundary >= right_boundary:
        left_boundary = max(0, center - 1000)
        right_boundary = min(width - 1, center + 1000)

    # 6. 背景均值合成逻辑
    bg_vals = []
    if left_bg_mean is not None:
        bg_vals.append(left_bg_mean)
    if right_bg_mean is not None:
        bg_vals.append(right_bg_mean)

    if bg_vals:
        bg_mean = float(np.mean(bg_vals))
    else:
        bg_mean = zsz_Constants.Normal_BACKGROUND

    # 7. 构造膜区域 mask
    mask = np.zeros_like(gray, dtype=bool)
    mask[:, left_boundary : right_boundary + 1] = True

    return mask, (left_boundary, right_boundary), bg_mean
