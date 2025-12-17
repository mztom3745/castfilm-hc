import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image

from detect_core.castfilm_detector import CastFilmDefectDetector
from detect_core.defect_classifier import DefectClassifier
from detect_core.defect_config import DefectConfig


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """安全读取图片，支持中文路径"""
    try:
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, flags)
            return img
    except Exception as e:
        print(f"⚠️ 无法读取文件: {os.path.basename(path)}, 错误: {e}")
        return None


def save_patch_image(patch: np.ndarray, save_path: str) -> bool:
    """保存缺陷小图，自动创建目录（PIL 支持中文路径）"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        if patch.ndim == 2:
            img = Image.fromarray(patch, mode="L")
        else:
            if patch.ndim == 3 and patch.shape[2] == 1:
                patch = patch[:, :, 0]
                img = Image.fromarray(patch, mode="L")
            else:
                img = Image.fromarray(patch, mode="RGB")

        img.save(save_path)
        return True
    except Exception as e:
        print(f"❌ 保存失败: {save_path}，错误: {e}")
        return False


def _format_space_table(headers: List[str], rows: List[List[str]], pad: int = 2) -> str:
    """
    用空格做等宽对齐的表格：
    - 第一列左对齐，其余列右对齐（更像数字表）
    """
    if not headers:
        return ""

    # 计算每列最大宽度
    col_count = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(col_count):
            widths[i] = max(widths[i], len(str(r[i])))

    lines = []

    # 表头
    head_cells = []
    for i, h in enumerate(headers):
        if i == 0:
            head_cells.append(h.ljust(widths[i]))
        else:
            head_cells.append(h.rjust(widths[i]))
    sep = " " * pad
    lines.append(sep.join(head_cells))

    # 分割线
    total_width = sum(widths) + pad * (col_count - 1)
    lines.append("-" * total_width)

    # 内容
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            v = str(v)
            if i == 0:
                cells.append(v.ljust(widths[i]))
            else:
                cells.append(v.rjust(widths[i]))
        lines.append(sep.join(cells))

    return "\n".join(lines)


def classify_all_images_in_folder(input_folder: str, output_root: str):
    """
    读取 input_folder 中的所有图片：
      1) 分割 defects
      2) 分类
      3) 按 尺寸/类别 存 patch 到输出目录
      4) 写全局汇总 txt（空格等宽表格）
    """
    print(f"🔍 输入文件夹: {input_folder}")
    if not os.path.isdir(input_folder):
        print(f"[ERROR] 输入路径不是有效文件夹：{input_folder}")
        return

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    image_files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if not image_files:
        print("[WARN] 该文件夹下未找到任何图片文件")
        return

    # ✅ 每个输入文件夹，建立一个独立输出子目录
    input_name = os.path.basename(os.path.normpath(input_folder))
    base_output = os.path.abspath(os.path.join(output_root, input_name))
    os.makedirs(base_output, exist_ok=True)

    print(f"📤 输出目录: {base_output}")

    detector = CastFilmDefectDetector()
    classifier = DefectClassifier()

    # 全局统计：尺寸 -> 类别 -> 数量
    global_stats: Dict[str, Dict[str, int]] = {}

    total_images = len(image_files)
    print(f"📂 共找到 {total_images} 张图片，将逐一处理 ...")

    for idx_img, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
        image_stem, _ = os.path.splitext(image_name)
        print(f"\n========================")
        print(f"📸 [{idx_img}/{total_images}] 处理图片: {image_name}")

        img_gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        defects, left_edge_x, right_edge_x = detector.detect_defects_fast(img_gray)
        print(f"   ➤ 分割完成，检测到缺陷数量: {len(defects)}")
        if not defects:
            continue

        classify_results = classifier.classify_defects_batch(
            [img_gray],
            [defects],
        )
        result: Dict = classify_results[0] if classify_results else {}
        cut_images = result.get("cut_images", [])
        defect_infos = result.get("defect_infos", [])
        if not cut_images or not defect_infos:
            print("   ⚠️ 分类结果为空，跳过。")
            continue

        saved_patches = 0
        total_patches = 0

        for idx, (patch, info) in enumerate(zip(cut_images, defect_infos), start=1):
            defect_class = info.get("defect_class", "其它")
            size_index = info.get("size_index", 0)
            if 0 <= size_index < len(DefectConfig.SIZE_LIST):
                size_name = DefectConfig.SIZE_LIST[size_index]
            else:
                size_name = "未知尺寸"

            # 全局统计更新
            global_stats.setdefault(size_name, {})
            global_stats[size_name][defect_class] = global_stats[size_name].get(defect_class, 0) + 1

            # 存图：output/输入名/尺寸/类别/xxx.png
            subdir = os.path.join(base_output, size_name, defect_class)
            filename = f"{image_stem}_defect_{idx:03d}.png"
            save_path = os.path.join(subdir, filename)

            total_patches += 1
            if save_patch_image(patch, save_path):
                saved_patches += 1

        print(f"   💾 小图保存统计: 成功 {saved_patches} / 共 {total_patches} 张")

    # ===== 写全局汇总 txt（空格等宽对齐）=====
    summary_path = os.path.join(base_output, "classification_summary.txt")
    size_order = list(DefectConfig.SIZE_LIST)

    for s in size_order:
        global_stats.setdefault(s, {})

    # ✅ 固定四类（顺序也固定）
    class_order = ["黑点", "晶点", "纤维", "其它"]

    # ✅ 防御：确保所有尺寸下这四类键都存在（没有就补 0）
    for s in size_order:
        global_stats.setdefault(s, {})
        for cls in class_order:
            global_stats[s].setdefault(cls, 0)

    # 组装表格 rows：每行一个类别 + TOTAL
    headers = ["类别\\尺寸"] + size_order + ["TOTAL"]
    rows: List[List[str]] = []

    grand_total = 0
    for cls in class_order:
        row_total = 0
        row_vals = []
        for s in size_order:
            cnt = int(global_stats[s].get(cls, 0))
            row_vals.append(str(cnt))
            row_total += cnt
        grand_total += row_total
        rows.append([cls] + row_vals + [str(row_total)])

    col_totals = []
    for s in size_order:
        col_totals.append(str(sum(int(global_stats[s].get(cls, 0)) for cls in class_order)))
    rows.append(["TOTAL"] + col_totals + [str(grand_total)])


    table_text = _format_space_table(headers, rows, pad=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CastFilm 分类结果全局汇总（类别 × 尺寸）\n")
        f.write(f"输入文件夹: {os.path.abspath(input_folder)}\n")
        f.write(f"输出文件夹: {base_output}\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_text + "\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"全部图像缺陷总数（裁剪总数）: {grand_total}\n")

    print(f"\n✅ 汇总已写入: {summary_path}")


if __name__ == "__main__":
    ROOT_INPUT_FOLDER = r"D:\castfilm-hc\data"
    ROOT_OUTPUT_FOLDER = r"./single_image_classify_output"

    classify_all_images_in_folder(
        input_folder=ROOT_INPUT_FOLDER,
        output_root=ROOT_OUTPUT_FOLDER,
    )
