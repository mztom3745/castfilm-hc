import os
import time
import gc

import cv2
import numpy as np
import torch

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


# ------------------------
# 1. 统计更新
# ------------------------
def process_dataset(classify_result, count):
    """
    仅更新统计计数器：
    对于每个缺陷：
        - 取 defect_class（黑点/晶点/纤维/其它）
        - 取 size_index → SIZE_LIST 名称
        - 在 count[defect_class][size_name] 里 +1
    """
    cut_images = classify_result.get("cut_images", [])
    defect_infos = classify_result.get("defect_infos", [])
    for _, defect_info in zip(cut_images, defect_infos):
        defect_class = defect_info.get("defect_class", "其它")
        size_index = defect_info.get("size_index", 0)
        if 0 <= size_index < len(DefectConfig.SIZE_LIST):
            size_name = DefectConfig.SIZE_LIST[size_index]
            # 兜底：如果分类器返回了未预设的类别名，自动归到“其它”
            if defect_class not in count:
                defect_class = "其它"
            count[defect_class][size_name] += 1


# ------------------------
# 2. 单个子文件夹处理：分割 + 分类 + 统计
#    👉 完全单线程顺序处理
# ------------------------
def process_single_subfolder(
    subfolder_path,
    batch_size=6,
    queue_maxsize=200,
    classifier=None,
):
    """
    处理单个子文件夹（单线程版本）：
        1) 从子文件夹读取所有图像
        2) 顺序处理每一张：分割 -> 分类 -> 统计
    返回： (子文件夹路径, 统计字典)
    """
    print(f"\n📂 开始处理子文件夹：{os.path.basename(subfolder_path)}")

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        os.path.join(subfolder_path, f)
        for f in sorted(os.listdir(subfolder_path))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if not image_files:
        print(f"[WARN] 未找到图像文件：{os.path.basename(subfolder_path)}")
        return subfolder_path, {}

    # 初始化统计
    categories = ["黑点", "晶点", "纤维", "其它"]
    count = {cat: {size: 0 for size in DefectConfig.SIZE_LIST} for cat in categories}

    # ✅ 在当前进程初始化一次检测器
    detector = CastFilmDefectDetector()

    # ---------------- 顺序分割 + 分类 ----------------
    print(f"🚀 启动分割+分类任务（单线程），共 {len(image_files)} 张")
    start_all = time.perf_counter()

    processed = 0
    for idx, image_path in enumerate(image_files, 1):
        img_gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"⚠️ 无法读取文件: {os.path.basename(image_path)}")
            continue

        try:
            # 分割
            seg_start = time.perf_counter()
            defects, left_edge_x, right_edge_x = detector.detect_defects_fast(img_gray)
            seg_time = time.perf_counter() - seg_start

            # 🔥 每张图输出缺陷数量
            print(f"[缺陷数量] {os.path.basename(image_path)}: {len(defects)}")

            # 分类（仍然按 1 张图一送；如果你后续要攒 batch，再改这里）
            cls_start = time.perf_counter()
            classify_results = classifier.classify_defects_batch([img_gray], [defects])
            cls_time = time.perf_counter() - cls_start

            # 统计
            for result in classify_results:
                process_dataset(result, count)

            processed += 1
            elapsed = time.perf_counter() - start_all
            avg_time = elapsed / processed if processed > 0 else 0.0

            if idx % 10 == 0 or idx == len(image_files):
                print(
                    f"[进度] {idx}/{len(image_files)} | "
                    f"总耗时 {elapsed:.1f}s | 平均 {avg_time:.3f}s/张 | "
                    f"最近一张: 分割 {seg_time:.3f}s + 分类 {cls_time:.3f}s"
                )

        except Exception as e:
            print(f"⚠️ 分割/分类阶段出错: {os.path.basename(image_path)}, 错误: {e}")
            continue

        # 可选：适当手动清一下
        # del img_gray, defects, classify_results
        # gc.collect()

    # ✅ 清理显存与内存
    torch.cuda.empty_cache()
    gc.collect()

    # 🔥 子文件夹缺陷总数
    total_defects = sum(count[cat][size] for cat in count for size in count[cat])
    print(f"📊 子文件夹缺陷总数: {total_defects}")

    print(f"✅ 子文件夹完成：{os.path.basename(subfolder_path)}")
    return subfolder_path, count


# ------------------------
# 3. 多子文件夹批量处理 + 汇总报告
#    👉 外层顺序遍历子文件夹
# ------------------------
def process_multi_subfolders(
    root_input_folder,
    root_output_folder,
    batch_size=6,
    max_workers=8,
    queue_maxsize=200,
    report_name_level=1,
    custom_name=None,
):
    """
    多文件夹批量检测（单线程版本）：
        - root_input_folder 下每个子目录依次处理
        - 不保存图像，只写一个汇总 txt 报告
    """
    print(f"🔍 启动多文件夹批量检测: {root_input_folder}")
    if not os.path.exists(root_input_folder):
        print(f"[ERROR] 输入路径不存在：{root_input_folder}")
        return

    os.makedirs(root_output_folder, exist_ok=True)

    # ✅ 动态报告命名逻辑
    path_parts = os.path.normpath(root_input_folder).split(os.sep)
    if custom_name:
        folder_name = custom_name
    elif len(path_parts) >= report_name_level:
        folder_name = path_parts[-report_name_level]
    else:
        folder_name = os.path.basename(os.path.normpath(root_input_folder))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"{folder_name}_summary_{timestamp}.txt"
    report_path = os.path.join(root_output_folder, report_filename)
    print(f"📄 汇总报告将保存为: {report_filename}")

    subfolders = [
        os.path.join(root_input_folder, d)
        for d in os.listdir(root_input_folder)
        if os.path.isdir(os.path.join(root_input_folder, d))
    ]
    if not subfolders:
        print(f"[ERROR] 没有子文件夹")
        return

    # ✅ 外层只初始化一次分类器
    defect_classifier = DefectClassifier()

    results = []
    for subfolder in subfolders:
        res = process_single_subfolder(
            subfolder_path=subfolder,
            batch_size=batch_size,
            queue_maxsize=queue_maxsize,
            classifier=defect_classifier,
        )
        results.append(res)

    # 🔥 计算所有子文件夹总缺陷数量
    grand_total = 0
    for _, count in results:
        if not count:
            continue
        grand_total += sum(count[cat][size] for cat in count for size in count[cat])

    # ✅ 写出汇总报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("多文件夹缺陷检测汇总报告（无光照检测）\n")
        f.write(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入根目录：{root_input_folder}\n")
        f.write("=" * 80 + "\n\n")

        for subfolder_path, count in results:
            f.write(f"文件夹: {os.path.basename(subfolder_path)}\n")
            if count:
                subfolder_total = sum(
                    count[cat][size] for cat in count for size in count[cat]
                )
                f.write(f"缺陷总数: {subfolder_total}\n")
                f.write("class_name " + " ".join(DefectConfig.SIZE_LIST) + "\n")
                for cat in ["黑点", "晶点", "纤维", "其它"]:
                    f.write(
                        f"{cat} "
                        + " ".join(str(count[cat][s]) for s in DefectConfig.SIZE_LIST)
                        + "\n"
                    )
            else:
                f.write("缺陷统计: 无（该文件夹无图像或处理失败）\n")
            f.write("-" * 80 + "\n\n")

        f.write(f"所有子文件夹总缺陷数量: {grand_total}\n")

    print(f"🎯 所有子文件夹总缺陷数量：{grand_total}")
    print(f"🎉 所有子文件夹处理完成！报告保存至 {report_path}")


if __name__ == "__main__":
    ROOT_INPUT_FOLDER = r"D:\castfilm-hc\data"
    ROOT_OUTPUT_FOLDER = r"./output_summary_reports"

    process_multi_subfolders(
        root_input_folder=ROOT_INPUT_FOLDER,
        root_output_folder=ROOT_OUTPUT_FOLDER,
        batch_size=4,          # 目前仍未实际用到（你后续要攒 batch 可用）
        max_workers=1,         # 单线程版本，这个参数无效，仅占位
        queue_maxsize=512,     # 目前未实际用到（占位）
        report_name_level=2,
        # custom_name="film_batchA"
    )
