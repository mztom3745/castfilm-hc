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
        print("[平滑] 空信号，直接返回空拷贝")
        return signal.copy()

    window = max(5, int(n * window_ratio))
    window = _ensure_odd(window)
    pad = window // 2

    print(
        f"[平滑] 信号长度 n={n}，窗口比例={window_ratio}，"
        f"实际窗口大小 window={window}，两侧填充像素 pad={pad}"
    )

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
    左侧逻辑（严格按照你说的）：

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

    print(
        f"[左侧] 粗略膜左边界 left_guess={left_guess} "
        f"（假设中间 ±{half_window} 为膜）"
    )
    print(
        f"[左侧] 搜索区间: [{search_start}, {search_end}] "
        f"（从粗略膜边界向更左侧找高梯度）"
    )

    side_idx = high_idx[(high_idx >= search_start) & (high_idx <= search_end)]
    side_idx_sorted = np.sort(side_idx)

    print(f"[左侧] 当前侧高梯度数量 = {side_idx_sorted.size}")
    if side_idx_sorted.size > 0:
        print(f"[左侧] 高梯度位置示例(前10个): {side_idx_sorted[:10]}")
        print(f"[左侧] 高梯度位置示例(后10个): {side_idx_sorted[-10:]}")

    if side_idx_sorted.size == 0:
        print("[左侧] 搜索区间内没有任何高梯度点 → 无法确定左侧边界")
        return None, None

    # 第一个高梯度：从粗略膜边界往外看，离 left_guess 最近的那个 → max(...)
    first_high = int(side_idx_sorted.max())
    print(
        f"[左侧] 距离粗略膜边界(left_guess={left_guess})最近的高梯度 first_high={first_high} "
        f"(left_guess-first_high={left_guess - first_high})"
    )

    # 最外侧的高梯度：越往 0 方向最小的索引
    last_high = int(side_idx_sorted.min())
    print(
        f"[左侧] 最靠外侧的高梯度 last_high={last_high} "
        f"(表示从 0 到 {last_high} 之间大致是膜-背景过渡附近)"
    )

    # 真正的膜左边界：在 first_high 基础上再向左偏移 5 像素
    left_boundary = max(0, first_high - 5)
    print(
        f"[左侧] 依据 first_high 往左偏移 5 像素 → "
        f"left_boundary = max(0, {first_high} - 5) = {left_boundary}"
    )

    # 背景区域：最后一个高梯度值的外部区域
    # 这里按你的语义，将 [0, last_high] 左侧理解为背景（或者说“膜外背景区域在 last_high 更外一段”）
    # 为了简单起见，这里仍取 [0, last_high] 作为背景块做均值检查
    bg_end_col = last_high - 1
    if bg_end_col <= 0:
        print(
            f"[左侧] 背景右边界 bg_end_col={bg_end_col} <= 0，"
            "没有可用的左侧背景区域"
        )
        return left_boundary, None

    bg_region = gray[:, 0:bg_end_col + 1]
    bg_mean_val = float(bg_region.mean())
    print(
        f"[左侧] 背景区域列区间: [0, {bg_end_col}]，"
        f"区域 shape={bg_region.shape}，灰度均值={bg_mean_val:.3f}"
    )

    if bg_mean_val < min_background_value:
        print(
            f"[左侧] 背景均值 {bg_mean_val:.3f} < 最小阈值 "
            f"{min_background_value:.3f} → 左侧边界视为无效"
        )
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

    print(
        f"[右侧] 粗略膜右边界 right_guess={right_guess} "
        f"（假设中间 ±{half_window} 为膜）"
    )

    # 右侧“往外找” ⇒ 只看 >= right_guess 的高梯度
    search_start = right_guess
    search_end = max_grad_idx

    print(
        f"[右侧] 搜索区间: [{search_start}, {search_end}] "
        f"（从粗略膜边界向更右侧找高梯度）"
    )

    side_idx = high_idx[(high_idx >= search_start) & (high_idx <= search_end)]
    side_idx_sorted = np.sort(side_idx)

    print(f"[右侧] 当前侧高梯度数量 = {side_idx_sorted.size}")
    if side_idx_sorted.size > 0:
        print(f"[右侧] 高梯度位置示例(前10个): {side_idx_sorted[:10]}")
        print(f"[右侧] 高梯度位置示例(后10个): {side_idx_sorted[-10:]}")

    if side_idx_sorted.size == 0:
        print("[右侧] 搜索区间内没有任何高梯度点 → 无法确定右侧边界")
        return None, None

    # 第一个高梯度：从粗略膜边界往外看，离 right_guess 最近的那个 → min(...)
    first_high = int(side_idx_sorted.min())
    print(
        f"[右侧] 距离粗略膜边界(right_guess={right_guess})最近的高梯度 first_high={first_high} "
        f"(first_high-right_guess={first_high - right_guess})"
    )

    # 最外侧的高梯度：越往右越大
    last_high = int(side_idx_sorted.max())
    print(
        f"[右侧] 最靠外侧的高梯度 last_high={last_high} "
        f"(表示从 {last_high+1} 到 {width-1} 之间大致是膜外背景区域)"
    )

    # 真正的膜右边界：在 first_high 基础上往右偏移 5 像素
    right_boundary = min(width - 1, first_high + 5)
    print(
        f"[右侧] 依据 first_high 往右偏移 5 像素 → "
        f"right_boundary = min({width-1}, {first_high} + 5) = {right_boundary}"
    )

    # 背景区域：最后一个高梯度值的外部区域 → [last_high+1, width-1]
    bg_start_col = last_high + 1
    if bg_start_col >= width:
        print(
            f"[右侧] 背景起始列 bg_start_col={bg_start_col} >= 宽度 {width}，"
            "没有可用的右侧背景区域"
        )
        return right_boundary, None

    bg_region = gray[:, bg_start_col:width]
    bg_mean_val = float(bg_region.mean())
    print(
        f"[右侧] 背景区域列区间: [{bg_start_col}, {width-1}]，"
        f"区域 shape={bg_region.shape}，灰度均值={bg_mean_val:.3f}"
    )

    if bg_mean_val < min_background_value:
        print(
            f"[右侧] 背景均值 {bg_mean_val:.3f} < 最小阈值 "
            f"{min_background_value:.3f} → 右侧边界视为无效"
        )
        return None, None

    return right_boundary, bg_mean_val


def build_membrane_mask_grad(
    gray: np.ndarray,
    ratio: float = 0.1,
    smoothing_ratio: float = 0.005,
    background_percentile: float = 0.995,
    drop_ratio: float = 0.03,
    min_run_ratio: float = 0.002,
    edge_clip_ratio: float = 0.05,
    min_background_value: float = 200.0,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    基于梯度 + “中间 10000 像素视为膜” + “从中间向两侧 5000 像素往外找高梯度”的膜边界检测版本。
    """

    gray = np.asarray(gray, dtype=np.float32)
    h, width = gray.shape

    print("\n================= build_membrane_mask_grad 调试开始 =================")
    print(f"[输入] 灰度图尺寸 = H={h}, W={width}")

    if width < 2:
        print("[输入] 宽度 < 2，无法做梯度 → 直接整幅视为膜区域，背景值设为 220")
        mask = np.ones_like(gray, dtype=bool)
        print("================= build_membrane_mask_grad 调试结束 =================\n")
        return mask, (0, width - 1), 220.0

    # 1. 列均值 + 平滑
    column_means = gray.mean(axis=0)
    print(f"[步骤1] 列均值长度 = {column_means.size} (应该等于宽度 {width})")

    smoothed = smooth_signal(column_means, window_ratio=smoothing_ratio)

    # 2. 一阶梯度及绝对值
    grad = np.diff(smoothed)          # 长度 = width - 1
    grad_abs = np.abs(grad)
    print(
        f"[步骤2] 梯度长度={grad_abs.size} (应为 W-1={width-1})，"
        f"梯度绝对值: 平均={grad_abs.mean():.3f}, 最大={grad_abs.max():.3f}"
    )

    # 3. 用分位数选高梯度阈值
    perc = float(np.clip(background_percentile * 100.0, 0.0, 100.0))
    thr = float(np.percentile(grad_abs, perc))
    high_idx = np.where(grad_abs >= thr)[0]  # index 范围 [0, width-2]

    print(
        f"[步骤3] 使用分位数 background_percentile={background_percentile} → "
        f"百分位 perc={perc:.2f}，高梯度阈值 thr={thr:.3f}"
    )
    print(f"[步骤3] 满足 |grad|>=thr 的高梯度索引数量 = {high_idx.size}")
    if high_idx.size > 0:
        print(f"[步骤3] 高梯度索引示例(前10个): {np.sort(high_idx)[:10]}")
        print(f"[步骤3] 高梯度索引示例(后10个): {np.sort(high_idx)[-10:]}")

    # 4. “中间 10000 像素是膜” + 从中间向两侧 5000 像素往外找高梯度
    center = width // 2
    half_window = 5000  # 即你说的“最中间的 10000 像素都是膜，中间±5000”
    print(
        f"[步骤4] 假定中间 10000 像素为膜 → center={center}, "
        f"粗略膜区 ≈ [{max(0, center-half_window)}, {min(width-1, center+half_window)}]"
    )

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

    print(f"[步骤4] 左侧返回: left_boundary={left_boundary}, left_bg_mean={left_bg_mean}")
    print(f"[步骤4] 右侧返回: right_boundary={right_boundary}, right_bg_mean={right_bg_mean}")

    # 5. 异常处理：没有找到某一侧边界 → 使用 center ± half_window 回退
    if left_boundary is None:
        left_boundary = max(0, center - half_window)
        print(f"[修正] 左侧没有合法边界 → 回退为 center-half_window={left_boundary}")
    if right_boundary is None:
        right_boundary = min(width - 1, center + half_window)
        print(f"[修正] 右侧没有合法边界 → 回退为 center+half_window={right_boundary}")

    # 防止 left >= right 的极端情况
    if left_boundary >= right_boundary:
        print(
            f"[修正] 出现 left_boundary({left_boundary}) >= right_boundary({right_boundary})，"
            "回退为 center±1000"
        )
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
        print(f"[步骤6] 有效背景均值列表 = {bg_vals} → 取平均 bg_mean={bg_mean:.3f}")
    else:
        bg_mean = 220.0
        print(
            "[步骤6] 左右两侧均无有效背景均值 → 使用默认背景值 bg_mean=220.0"
        )

    # 7. 构造膜区域 mask
    mask = np.zeros_like(gray, dtype=bool)
    mask[:, left_boundary : right_boundary + 1] = True

    print(
        f"[步骤7] 最终膜区域列范围: [{left_boundary}, {right_boundary}] "
        f"(宽度约 {right_boundary - left_boundary + 1} 像素)"
    )
    print(
        f"[结果] 左边界={left_boundary}, 右边界={right_boundary}, "
        f"背景均值 bg_mean={bg_mean:.3f}"
    )
    print("================= build_membrane_mask_grad 调试结束 =================\n")

    return mask, (left_boundary, right_boundary), bg_mean
