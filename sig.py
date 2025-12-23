import os
import cv2
import numpy as np
from typing import Dict, List
from PIL import Image  # 仍保留，避免你后面可能还用；本版本不再保存小图

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


def _format_space_table(headers: List[str], rows: List[List[str]], pad: int = 2) -> str:
    """
    用空格做等宽对齐的表格：
    - 第一列左对齐，其余列右对齐（更像数字表）
    """
    if not headers:
        return ""

    col_count = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(col_count):
            widths[i] = max(widths[i], len(str(r[i])))

    sep = " " * pad
    lines = []

    # 表头
    head_cells = []
    for i, h in enumerate(headers):
        head_cells.append(h.ljust(widths[i]) if i == 0 else h.rjust(widths[i]))
    lines.append(sep.join(head_cells))

    # 分割线
    total_width = sum(widths) + pad * (col_count - 1)
    lines.append("-" * total_width)

    # 内容
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            v = str(v)
            cells.append(v.ljust(widths[i]) if i == 0 else v.rjust(widths[i]))
        lines.append(sep.join(cells))

    return "\n".join(lines)


def classify_all_images_in_folder_report_only(input_folder: str, output_root: str):
    """
    读取 input_folder 中的所有图片：
      1) 分割 defects
      2) 分类
      3) 只做全局统计
      4) 写全局汇总 txt（空格等宽表格）
    不保存任何缺陷小图。
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

    # 每个输入文件夹，建立一个独立输出子目录（只放报告）
    input_name = os.path.basename(os.path.normpath(input_folder))
    base_output = os.path.abspath(os.path.join(output_root, input_name))
    os.makedirs(base_output, exist_ok=True)

    print(f"📤 输出目录（仅报告）: {base_output}")

    detector = CastFilmDefectDetector()
    classifier = DefectClassifier()

    # 全局统计：尺寸 -> 类别 -> 数量
    global_stats: Dict[str, Dict[str, int]] = {}

    total_images = len(image_files)
    print(f"📂 共找到 {total_images} 张图片，将逐一处理 ...")

    for idx_img, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
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
        defect_infos = result.get("defect_infos", [])
        if not defect_infos:
            print("   ⚠️ 分类结果为空，跳过。")
            continue

        # 只统计，不保存 patch
        for info in defect_infos:
            defect_class = info.get("defect_class", "其它")
            size_index = info.get("size_index", 0)
            if 0 <= size_index < len(DefectConfig.SIZE_LIST):
                size_name = DefectConfig.SIZE_LIST[size_index]
            else:
                size_name = "未知尺寸"

            global_stats.setdefault(size_name, {})
            global_stats[size_name][defect_class] = global_stats[size_name].get(defect_class, 0) + 1

    # ===== 写全局汇总 txt（空格等宽对齐）=====
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        base_output,
        f"classification_summary_{timestamp}.txt"
    )
    size_order = list(DefectConfig.SIZE_LIST)
    for s in size_order:
        global_stats.setdefault(s, {})

    class_order = ["黑点", "晶点", "纤维", "其它"]
    for s in size_order:
        for cls in class_order:
            global_stats[s].setdefault(cls, 0)

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
    ROOT_INPUT_FOLDER = r"C:\Users\WINDOWS\Desktop\data\815\2025年08月15日10时35分18秒"
    ROOT_OUTPUT_FOLDER = r"./single_image_classify_output"

    classify_all_images_in_folder_report_only(
        input_folder=ROOT_INPUT_FOLDER,
        output_root=ROOT_OUTPUT_FOLDER,
    )
