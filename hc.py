import os
import time
import gc
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import torch

from detect_core.castfilm_detector import CastFilmDefectDetector
from detect_core.defect_classifier import DefectClassifier
from detect_core.defect_config import DefectConfig


# ------------------------
# 1. 全局检测器初始化函数（供进程池共享）
# ------------------------
_detector = None


def _init_detector():
    """在每个子进程里初始化一次 CastFilm 检测器"""
    global _detector
    _detector = CastFilmDefectDetector()


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


def _segment_one_image(image_path):
    """
    子进程分割任务：
    1) 读取灰度图
    2) 调用 CastFilmDefectDetector.detect_defects_fast
    这里只保留缺陷框，不做任何保存/前端交互
    """
    try:
        global _detector
        image_np_gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
        if image_np_gray is None:
            return False, image_path, []

        defects, left_edge_x, right_edge_x = _detector.detect_defects_fast(image_np_gray)
        return True, image_path, defects
    except Exception as e:
        print(f"⚠️ 分割阶段出错: {os.path.basename(image_path)}, 错误: {e}")
        return False, image_path, []


# ------------------------
# 2. 分类 + 统计
# ------------------------
def process_dataset(classify_result, count):
    """
    仅更新统计计数器：
    对于每个缺陷：
        - defect_class（黑点/晶点/纤维/其它）
        - size_index → SIZE_LIST 名称
        - count[defect_class][size_name] += 1
    """
    cut_images = classify_result.get("cut_images", [])
    defect_infos = classify_result.get("defect_infos", [])
    for _, defect_info in zip(cut_images, defect_infos):
        defect_class = defect_info.get("defect_class", "其它")
        size_index = defect_info.get("size_index", 0)
        if 0 <= size_index < len(DefectConfig.SIZE_LIST):
            size_name = DefectConfig.SIZE_LIST[size_index]
            # 兜底：分类器如果给了不在预设里的类名，归入“其它”
            if defect_class not in count:
                defect_class = "其它"
            count[defect_class][size_name] += 1


def _safe_qsize(q):
    try:
        return q.qsize()
    except Exception:
        return -1


def classification_worker(result_queue: "queue.Queue", classifier, count, batch_size=6):
    """
    分类线程：
        - 从队列 result_queue 取 (image_path, defects)
        - 读灰度图，按 batch 做 classify_defects_batch
        - 用 process_dataset 更新统计
    队列里收到 None 时退出。
    """
    batch_images, batch_defects = [], []
    processed_count = 0
    cls_start = time.perf_counter()

    while True:
        item = result_queue.get()  # 阻塞等待
        if item is None:
            break

        image_path, defects = item
        image_np_gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
        if image_np_gray is None:
            continue

        batch_images.append(image_np_gray)
        batch_defects.append(defects)

        if len(batch_images) >= batch_size:
            t0 = time.perf_counter()
            classify_results = classifier.classify_defects_batch(batch_images, batch_defects)
            t1 = time.perf_counter()

            processed_count += len(batch_images)
            elapsed = t1 - cls_start
            avg_time = elapsed / processed_count if processed_count else 0.0
            speed = processed_count / elapsed if elapsed > 0 else 0.0
            qn = _safe_qsize(result_queue)

            print(
                f"[分类线程] 已分类 {processed_count} 张 | "
                f"本批耗时 {(t1 - t0):.3f}s | "
                f"平均 {avg_time:.3f}s/张 | "
                f"速度 {speed:.2f} 张/秒 | "
                f"队列长度 {qn if qn >= 0 else 'NA'}"
            )

            for result in classify_results:
                process_dataset(result, count)

            batch_images, batch_defects = [], []

    # 处理剩余未满 batch 的内容
    if batch_images:
        classify_results = classifier.classify_defects_batch(batch_images, batch_defects)
        for result in classify_results:
            process_dataset(result, count)


# ------------------------
# 3. 单个子文件夹处理：分割 + 分类 + 统计（无光照检测）
# ------------------------
def process_single_subfolder(
    subfolder_path,
    batch_size=6,
    queue_maxsize=200,
    classifier=None,
    executor=None,
):
    """
    处理单个子文件夹：
        1) 从子文件夹读取所有图像（只分割+分类，不保存）
        2) 用进程池做分割；用分类线程批量分类；更新尺寸统计
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

    # 初始化统计：每个类别 × 每个尺寸段
    categories = ["黑点", "晶点", "纤维", "其它"]
    count = {cat: {size: 0 for size in DefectConfig.SIZE_LIST} for cat in categories}

    # 启动分类线程（只做分类+统计）
    result_queue = queue.Queue(maxsize=queue_maxsize)
    cls_thread = threading.Thread(
        target=classification_worker,
        args=(result_queue, classifier, count, batch_size),
        daemon=True,
    )
    cls_thread.start()

    # ---------------- 分割部分 ----------------
    print(f"🚀 启动分割任务，共 {len(image_files)} 张")
    seg_start_all = time.perf_counter()
    completed = 0

    futures = [executor.submit(_segment_one_image, path) for path in image_files]
    for f in as_completed(futures):
        ok, image_path, defects = f.result()
        completed += 1

        if completed % 10 == 0 or completed == len(futures):
            elapsed = time.perf_counter() - seg_start_all
            avg_time = elapsed / completed if completed else 0.0
            print(f"[分割进度] {completed}/{len(futures)} | {elapsed:.1f}s | {avg_time:.3f}s/张")

        if not ok:
            continue

        # queue.Queue 会在满时阻塞，不用你手动 sleep 轮询
        result_queue.put((image_path, defects))

    # 通知分类线程结束
    result_queue.put(None)
    cls_thread.join()

    # 清理显存与内存
    torch.cuda.empty_cache()
    gc.collect()

    # 子文件夹缺陷总数（可打印）
    sub_total = sum(count[cat][size] for cat in count for size in count[cat])
    print(f"📊 子文件夹缺陷总数: {sub_total}")
    print(f"✅ 子文件夹完成：{os.path.basename(subfolder_path)}")

    return subfolder_path, count


# ------------------------
# 4. 多子文件夹批量处理 + 汇总报告（无光照检测）
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
    多文件夹批量检测（仅分割+分类+统计）：
        - root_input_folder 下每个子目录一个“批次”
        - 不保存图像，不与前端交互，只写一个汇总 txt 报告
    """
    print(f"🔍 启动多文件夹批量检测: {root_input_folder}")
    if not os.path.exists(root_input_folder):
        print(f"[ERROR] 输入路径不存在：{root_input_folder}")
        return

    os.makedirs(root_output_folder, exist_ok=True)

    # 动态报告命名逻辑
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

    # 外层只初始化一次分类器和进程池
    defect_classifier = DefectClassifier()

    results = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_detector) as executor:
        for subfolder in subfolders:
            res = process_single_subfolder(
                subfolder_path=subfolder,
                batch_size=batch_size,
                queue_maxsize=queue_maxsize,
                classifier=defect_classifier,
                executor=executor,
            )
            results.append(res)

    # 计算总缺陷数
    grand_total = 0
    for _, count in results:
        if not count:
            continue
        grand_total += sum(count[cat][size] for cat in count for size in count[cat])

    # 写出汇总报告（纯文本，不保存任何图像）
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("多文件夹缺陷检测汇总报告（无光照检测）\n")
        f.write(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入根目录：{root_input_folder}\n")
        f.write("=" * 80 + "\n\n")

        for subfolder_path, count in results:
            f.write(f"文件夹: {os.path.basename(subfolder_path)}\n")
            if count:
                sub_total = sum(count[cat][size] for cat in count for size in count[cat])
                f.write(f"缺陷总数: {sub_total}\n")
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
    # 👇 按需修改你的输入/输出路径
    ROOT_INPUT_FOLDER = r"G:\NEW\20250815\data2"
    ROOT_OUTPUT_FOLDER = "./output_summary_reports"

    process_multi_subfolders(
        root_input_folder=ROOT_INPUT_FOLDER,
        root_output_folder=ROOT_OUTPUT_FOLDER,
        batch_size=4,
        max_workers=8,
        queue_maxsize=512,
        report_name_level=2,       # 例如用倒数第二层命名
        # custom_name="film_batchA"
    )
