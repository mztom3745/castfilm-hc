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
    """
    保存缺陷小图，自动创建目录（使用 PIL 支持中文路径）。
    返回是否保存成功。
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 转成 uint8
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        # 根据维度判断灰度 / 彩色
        if patch.ndim == 2:
            img = Image.fromarray(patch, mode="L")
        else:
            if patch.ndim == 3 and patch.shape[2] == 1:
                patch = patch[:, :, 0]
                img = Image.fromarray(patch, mode="L")
            else:
                img = Image.fromarray(patch, mode="RGB")

        img.save(save_path)
        print(f"✅ 保存成功: {save_path}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {save_path}，错误: {e}")
        return False


def classify_all_images_in_folder(input_folder: str, output_folder: str):
    """
    读取 input_folder 中的所有图片：
      1. 每张图分割得到 defects
      2. 做分类
      3. 按 “尺寸 / 类别” 的层级，把每个缺陷的小图保存到 output_folder
      4. 在终端打印每张图的分类统计结果
      5. 将所有图片的统计结果 + 全局汇总写入一个 txt 文件
    """
    print(f"🔍 输入文件夹: {input_folder}")
    if not os.path.isdir(input_folder):
        print(f"[ERROR] 输入路径不是有效文件夹：{input_folder}")
        return

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if not image_files:
        print("[WARN] 该文件夹下未找到任何图片文件")
        return

    base_output = os.path.abspath(output_folder)
    os.makedirs(base_output, exist_ok=True)

    # 初始化分割器 & 分类器（复用）
    detector = CastFilmDefectDetector()
    classifier = DefectClassifier()

    # 全局统计：尺寸 -> 类别 -> 数量
    global_stats: Dict[str, Dict[str, int]] = {}
    # 每张图的统计记录：(image_name, stats_dict, total_patches)
    per_image_records: List[Tuple[str, Dict[str, Dict[str, int]], int]] = []

    total_images = len(image_files)
    print(f"📂 共找到 {total_images} 张图片，将逐一处理 ...")

    for idx_img, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
        image_stem, _ = os.path.splitext(image_name)
        print(f"\n========================")
        print(f"📸 [{idx_img}/{total_images}] 处理图片: {image_name}")

        # 读灰度图
        img_gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[ERROR] 无法读取图片：{image_path}")
            continue

        # 1️⃣ 分割：得到缺陷框
        defects, left_edge_x, right_edge_x = detector.detect_defects_fast(img_gray)
        print(f"   ➤ 分割完成，检测到缺陷数量: {len(defects)}")

        if len(defects) == 0:
            print("   ⚠️ 没有检测到缺陷，跳过分类。")
            continue

        # 2️⃣ 分类
        classify_results = classifier.classify_defects_batch(
            [img_gray],  # batch 中只有这一张图
            [defects],   # 对应的缺陷框列表
        )

        result: Dict = classify_results[0] if classify_results else {}
        cut_images = result.get("cut_images", [])
        defect_infos = result.get("defect_infos", [])

        if not cut_images or not defect_infos:
            print("   ⚠️ 分类结果为空（cut_images 或 defect_infos 为空），跳过。")
            continue

        # 当前图片统计：尺寸 -> 类别 -> 数量
        img_stats: Dict[str, Dict[str, int]] = {}
        total_patches = 0
        saved_patches = 0

        print("   ➤ 开始按 尺寸 / 类别 存储缺陷小图 ...")

        for idx, (patch, info) in enumerate(zip(cut_images, defect_infos), start=1):
            defect_class = info.get("defect_class", "其它")
            size_index = info.get("size_index", 0)

            if 0 <= size_index < len(DefectConfig.SIZE_LIST):
                size_name = DefectConfig.SIZE_LIST[size_index]
            else:
                size_name = "未知尺寸"

            # —— 当前图片统计更新 —— #
            img_stats.setdefault(size_name, {})
            img_stats[size_name].setdefault(defect_class, 0)
            img_stats[size_name][defect_class] += 1

            # —— 全局统计更新 —— #
            global_stats.setdefault(size_name, {})
            global_stats[size_name].setdefault(defect_class, 0)
            global_stats[size_name][defect_class] += 1

            # 新目录结构：output/尺寸/类别/*.png
            subdir = os.path.join(base_output, size_name, defect_class)
            filename = f"{image_stem}_defect_{idx:03d}.png"
            save_path = os.path.join(subdir, filename)

            total_patches += 1
            if save_patch_image(patch, save_path):
                saved_patches += 1

        # 终端打印当前图片的统计
        print("\n📊 当前图像分类结果（按 尺寸 -> 类别）：")
        print(f"   图像: {image_name}")
        for size_name in sorted(img_stats.keys()):
            print(f"  ▸ 尺寸: {size_name}")
            for cls_name, cnt in img_stats[size_name].items():
                print(f"      - 类别: {cls_name:<4}  数量: {cnt}")

        print(f"   💾 小图保存统计: 成功 {saved_patches} / 共 {total_patches} 张")

        per_image_records.append((image_name, img_stats, total_patches))

        # ===== 写出整体统计到 txt =====
    summary_path = os.path.join(base_output, "classification_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CastFilm 分类结果全局汇总（类别 × 尺寸）\n")
        f.write(f"输入文件夹: {os.path.abspath(input_folder)}\n")
        f.write(f"输出文件夹: {base_output}\n")
        f.write("=" * 80 + "\n\n")

        # ✅ 固定尺寸顺序：直接用 DefectConfig.SIZE_LIST（就是你说的那 11 个）
        size_order = list(DefectConfig.SIZE_LIST)

        # ✅ 确保每个尺寸都存在（哪怕 0）
        for s in size_order:
            global_stats.setdefault(s, {})

        # ✅ 收集全局所有类别（所有尺寸里的 defect_class）
        all_classes = set()
        for s in size_order:
            all_classes.update(global_stats[s].keys())
        class_order = sorted(all_classes)  # 你如果想固定类别顺序，这里换成你的列表即可

        # —— 写表头（第一行：尺寸）——
        # 形式：类别\尺寸 | _25 | 25-50 | ... | O800 | TOTAL
        f.write("类别\\尺寸\t" + "\t".join(size_order) + "\tTOTAL\n")

        grand_total = 0

        # —— 每一行一个类别 —— 
        for cls in class_order:
            row_total = 0
            row_vals = []
            for s in size_order:
                cnt = int(global_stats[s].get(cls, 0))
                row_vals.append(str(cnt))
                row_total += cnt

            grand_total += row_total
            f.write(f"{cls}\t" + "\t".join(row_vals) + f"\t{row_total}\n")

        # —— 最后一行 TOTAL（每个尺寸的总数 + 全部总数）——
        col_totals = []
        for s in size_order:
            col_sum = sum(int(global_stats[s].get(cls, 0)) for cls in class_order)
            col_totals.append(str(col_sum))
        f.write("TOTAL\t" + "\t".join(col_totals) + f"\t{grand_total}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"全部图像缺陷总数（裁剪总数）: {grand_total}\n")


if __name__ == "__main__":
    # 👇 输入文件夹：里面直接是图片
    ROOT_INPUT_FOLDER = r"D:\castfilm-hc\data"

    # 👇 输出文件夹：按 尺寸 / 类别 组织
    ROOT_OUTPUT_FOLDER = r"./single_image_classify_output"

    classify_all_images_in_folder(
        input_folder=ROOT_INPUT_FOLDER,
        output_folder=ROOT_OUTPUT_FOLDER,
    )
