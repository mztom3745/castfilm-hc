# run_membrane_grad_batch.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from membrane_grad_core2 import build_membrane_mask_grad


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


def process_single_image(
    image_path: Path,
    out_dir: Path,
    smoothing_ratio: float = 0.005,
    background_percentile: float = 0.995,
    min_background_value: float = 200.0,
) -> None:
    ensure_directory(out_dir)

    print(f"\n📸 处理图像: {image_path.name}")

    # 1. 读取灰度图
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)
    h, w = gray.shape

    # 2. 调用基于梯度的膜检测函数
    mask, (left, right), bg_mean = build_membrane_mask_grad(
        gray,
        smoothing_ratio=smoothing_ratio,
        background_percentile=background_percentile,
        min_background_value=min_background_value,
    )

    print(f"   → left = {left}, right = {right}, width = {w}")
    print(f"   → 膜宽度 = {right - left + 1} 像素")
    print(f"   → 背景均值 bg_mean = {bg_mean:.3f}")

    # 3. 在原图上用不同颜色标记膜区域 & 背景区域
    #    膜区域 = True；背景区域 = False
    base = gray.clip(0, 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    # 颜色定义
    membrane_color = np.array([0, 255, 0], dtype=np.float32)   # 绿色：膜
    background_color = np.array([0, 0, 255], dtype=np.float32) # 蓝色：背景

    mem_mask = mask
    bg_mask = ~mask

    # 半透明混合: 0.5 原图 + 0.5 颜色
    rgb[mem_mask] = 0.5 * rgb[mem_mask] + 0.5 * membrane_color
    rgb[bg_mask] = 0.5 * rgb[bg_mask] + 0.5 * background_color

    rgb_uint8 = rgb.clip(0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(rgb_uint8)

    # 4. 保存结果图 & 简单的边界信息文本
    ensure_directory(out_dir)
    overlay_path = out_dir / f"{image_path.stem}_membrane_bg_overlay.png"
    overlay_img.save(overlay_path)

    info_path = out_dir / "membrane_info.txt"
    with info_path.open("w", encoding="utf-8") as f:
        f.write(f"image_name = {image_path.name}\n")
        f.write(f"height = {h}, width = {w}\n")
        f.write(f"left = {left}, right = {right}\n")
        f.write(f"membrane_width = {right - left + 1}\n")
        f.write(f"background_mean = {bg_mean:.6f}\n")

    print(f"   ✅ 膜+背景叠加图已保存: {overlay_path}")
    print(f"   ✅ 边界信息已保存: {info_path}")

    img.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="基于梯度的膜边界检测，对文件夹中所有图像绘制膜区域与背景区域。"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("./data"),
        help="输入图片文件夹（会递归遍历所有图片）。",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./membrane_grad_outputs"),
        help="输出结果根目录。",
    )
    p.add_argument(
        "--smoothing-ratio",
        type=float,
        default=0.005,
        help="平滑窗口比例 (window = max(5, int(width * smoothing_ratio)))",
    )
    p.add_argument(
        "--background-percentile",
        type=float,
        default=0.995,
        help="用于选取高梯度阈值的分位数（0~1，例如 0.995）。",
    )
    p.add_argument(
        "--min-background-value",
        type=float,
        default=200.0,
        help="背景最低灰度均值阈值，低于该值视为该侧无有效边界。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    files = list(iter_image_files(args.input))
    if not files:
        print(f"⚠ 未在 {args.input} 下找到任何图片")
        return

    print(f"🔍 共找到 {len(files)} 张图片，将逐一进行膜+背景划分 ...")

    for img_path in files:
        out_dir = args.output_dir / img_path.stem
        process_single_image(
            image_path=img_path,
            out_dir=out_dir,
            smoothing_ratio=args.smoothing_ratio,
            background_percentile=args.background_percentile,
            min_background_value=args.min_background_value,
        )

    print("\n🎉 全部图像处理完成，可在输出目录查看各自子文件夹中的 overlay 图与信息。")


if __name__ == "__main__":
    main()
