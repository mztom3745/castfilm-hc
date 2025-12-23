# test_size_index.py
import math

def compute_size_index_idx_minus_1(area_pixels: float, bins, labels) -> int:
    """你的当前实现：idx-1"""
    if not bins or not labels:
        return 0
    for idx, boundary in enumerate(bins):
        if area_pixels < boundary:
            return max(0, idx - 1)
    return len(labels) - 1

def compute_size_index_strict(area_pixels: float, bins, labels) -> int:
    """
    严格对齐你注释区间的实现：
    bins[i] <= area < bins[i+1] -> i
    适用于 len(bins) = len(labels)+1 (你现在就是这种结构)
    """
    if not bins or not labels:
        return 0
    # 兜底：如果 bins 比 labels 少，不严格处理，直接回退最后一档
    n = min(len(labels), len(bins) - 1)
    for i in range(n):
        left = bins[i]
        right = bins[i + 1]
        if left <= area_pixels < right:
            return i
    return n - 1 if n > 0 else 0

def diameter_um_to_pixel_area(d_um: float, um_per_pixel: float) -> float:
    """把等效圆直径(um)转换为像素面积阈值：pi*(r/um_per_pixel)^2"""
    if math.isinf(d_um):
        return float("inf")
    r_um = d_um / 2.0
    r_px = r_um / um_per_pixel
    return math.pi * (r_px ** 2)

def main():
    # ===== 这里按你的工程路径导入 =====
    # 如果你脚本放在项目根目录，一般这样导入：
    from detect_core.defect_config import DefectConfig
    # 如果你的路径不同，按你的实际工程改 import 即可

    um_per_pixel = float(DefectConfig.UM_PER_PIXEL)
    dia_bins_um = list(DefectConfig.DIAMETER_BINS_UM)
    bins = list(map(float, DefectConfig.PIXEL_AREA_BINS))
    labels = list(DefectConfig.SIZE_LIST)

    print("=== Config Dump ===")
    print("UM_PER_PIXEL:", um_per_pixel)
    print("DIAMETER_BINS_UM len:", len(dia_bins_um), "->", dia_bins_um)
    print("PIXEL_AREA_BINS len:", len(bins))
    print("SIZE_LIST len:", len(labels))
    print("First few PIXEL_AREA_BINS:", bins[:6], "... last:", bins[-3:])
    print("Labels:", labels)
    print()

    # sanity: bins 应该是单调非降（最后一个 inf）
    monotonic = all(bins[i] <= bins[i+1] for i in range(len(bins)-1))
    print("Bins monotonic non-decreasing?", monotonic)
    if not monotonic:
        for i in range(len(bins)-1):
            if bins[i] > bins[i+1]:
                print("  Non-monotonic at", i, bins[i], ">", bins[i+1])
        print()

    # ===== 测试直径点（你也可以自行增删）=====
    test_diams_um = [0, 10, 20, 24.9, 25, 30, 40, 49.9, 50,
                     60, 90, 100, 150, 199.9, 200, 250, 300,
                     399.9, 400, 600, 799.9, 800, 900, 1200]

    print("=== Test Results (using equivalent-circle pixel area) ===")
    header = f"{'d_um':>7} | {'area_px':>10} | {'idx-1':>5} {'label':>7} | {'strict':>6} {'label':>7} | OK?"
    print(header)
    print("-" * len(header))

    mismatch = 0
    for d in test_diams_um:
        area_px = diameter_um_to_pixel_area(float(d), um_per_pixel)

        idx1 = compute_size_index_idx_minus_1(area_px, bins, labels)
        idx2 = compute_size_index_strict(area_px, bins, labels)

        label1 = labels[idx1] if 0 <= idx1 < len(labels) else "OUT"
        label2 = labels[idx2] if 0 <= idx2 < len(labels) else "OUT"

        ok = (idx1 == idx2)
        if not ok:
            mismatch += 1

        print(f"{d:7.1f} | {area_px:10.1f} | {idx1:5d} {label1:>7} | {idx2:6d} {label2:>7} | {'✅' if ok else '❌'}")

    print()
    if mismatch == 0:
        print("✅ 结果：idx-1 写法与严格区间法完全一致（在本测试点上未发现分档差异）。")
    else:
        print(f"❌ 结果：发现 {mismatch} 处分档不一致。通常意味着 idx-1 写法与注释区间不严格对齐。")
        print("   重点关注：_25 与 25-50 的边界（25um/50um附近）是否符合你期望。")

    # ===== 额外：随机抽样看整体一致性（可选）=====
    import random
    random.seed(0)
    rand_mismatch = 0
    for _ in range(2000):
        # 在 0~1200um 随机抽直径
        d = random.random() * 1200.0
        area_px = diameter_um_to_pixel_area(d, um_per_pixel)
        if compute_size_index_idx_minus_1(area_px, bins, labels) != compute_size_index_strict(area_px, bins, labels):
            rand_mismatch += 1
    print(f"\nRandom check: mismatches = {rand_mismatch} / 2000")

if __name__ == "__main__":
    main()
