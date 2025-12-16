from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------
# 辅助函数
# --------------------------
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(root: Path):
    """递归遍历所有图像文件"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    root = Path(root)
    if root.is_file() and root.suffix.lower() in exts:
        yield root
        return

    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def _ensure_odd(v: int) -> int:
    return v + 1 if v % 2 == 0 else v


def smooth_signal(signal: np.ndarray, window_ratio: float = 0.003) -> np.ndarray:
    """
    对信号进行平滑卷积，使用 edge-padding（复制信号两端值）避免 0 填充导致边缘失真。
    """
    n = signal.size
    window = max(5, int(n * window_ratio))
    window = _ensure_odd(window)

    pad = window // 2  # 例如 window=91, pad=45

    # 🔥 不使用0填充，使用 edge-padding 重复边缘值
    padded = np.pad(signal.astype(np.float32), pad_width=pad, mode='edge')

    # 平滑卷积
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(padded, kernel, mode="valid")  # valid 可返回正确长度：n

    return smoothed



# --------------------------
# 单张图像分析逻辑
# --------------------------
def analyze_image(
    image_path: Path,
    out_dir: Path,
    smoothing_ratio: float = 0.003,
    grad_percentile: float = 99.5,
):
    ensure_directory(out_dir)

    print(f"📸 分析: {image_path.name}")

    # 1. 读取灰度图
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)
    h, w = gray.shape

    # 2. 每列均值
    column_means = gray.mean(axis=0)

    # 3. 平滑
    smoothed = smooth_signal(column_means, smoothing_ratio)

    # 4. 梯度（diff）
    grad = np.diff(smoothed)
    grad_abs = np.abs(grad)

    # 5. 阈值：98 分位
    thr = float(np.percentile(grad_abs, grad_percentile))
    high_idx = np.where(grad_abs >= thr)[0]

    print(f"➡ 梯度阈值({grad_percentile}%) = {thr:.4f}, 高梯度点数量 = {high_idx.size}")

    # --------------------------
    # (A) 保存梯度 / 信号到 txt
    # --------------------------
    np.savetxt(out_dir / "grad.txt", grad, fmt="%.6f")
    np.savetxt(out_dir / "gradient_abs.txt", grad_abs, fmt="%.6f")
    np.savetxt(out_dir / "column_means.txt", column_means, fmt="%.6f")
    np.savetxt(out_dir / "smoothed.txt", smoothed, fmt="%.6f")

    # --------------------------
    # (B) 在原图上画高梯度红线
    # --------------------------
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    for i in high_idx:
        for c in (i, i + 1):
            if 0 <= c < w:
                rgb[:, c, 0] = 255
                rgb[:, c, 1] = 0
                rgb[:, c, 2] = 0

    overlay = Image.fromarray(rgb)
    overlay.save(out_dir / f"{image_path.stem}_overlay.png")

    # --------------------------
    # (C) 调试曲线图
    # --------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    x_cols = np.arange(w)
    x_grad = np.arange(w - 1)

    # 原始均值 & 平滑
    axes[0].plot(x_cols, column_means, label="column_means")
    axes[0].plot(x_cols, smoothed, label="smoothed")
    axes[0].legend()
    axes[0].set_title("Column means & smoothed")

    # 梯度 + 阈值
    axes[1].plot(x_grad, grad_abs, label="|grad|")
    axes[1].axhline(thr, color="r", linestyle="--", label=f"{grad_percentile}%")
    axes[1].legend()
    axes[1].set_title("Gradient magnitude")

    # 把高梯度位置画在 smoothed 曲线上
    axes[2].plot(x_cols, smoothed, label="smoothed")
    for ii in high_idx:
        axes[2].axvline(ii, color="r", alpha=0.4)
        axes[2].axvline(ii + 1, color="r", alpha=0.4)
    axes[2].legend()
    axes[2].set_title("High gradient positions on smoothed")

    plt.tight_layout()
    fig.savefig(out_dir / f"{image_path.stem}_grad_debug.png", dpi=150)
    plt.close(fig)

    img.close()
    print(f"✅ 完成 {image_path.name} → 输出目录: {out_dir}")


# --------------------------
# 主入口
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("./data"), help="输入图片文件夹")
    p.add_argument("--output-dir", type=Path, default=Path("./grad_debug_outputs"))
    p.add_argument("--smoothing-ratio", type=float, default=0.003)
    p.add_argument("--grad-percentile", type=float, default=99.5)
    return p.parse_args()


def main():
    args = parse_args()
    files = list(iter_image_files(args.input))

    if not files:
        print("⚠ 未找到图片")
        return

    print(f"🔍 共找到 {len(files)} 张图片")

    for img_path in files:
        out_dir = args.output_dir / img_path.stem
        analyze_image(
            image_path=img_path,
            out_dir=out_dir,
            smoothing_ratio=args.smoothing_ratio,
            grad_percentile=args.grad_percentile,
        )

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
