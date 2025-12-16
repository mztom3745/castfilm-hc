import asyncio
import base64
import datetime
import io
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Value, Queue
from queue import Empty

import aiofiles
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from camera.DalsaCamera2 import DalsaCamera2
from config.constant import *
from config.constant  import Border
from camera.DalsaCamera import DalsaCamera
from detect_core.castfilm_detector import CastFilmDefectDetector
from detect_core.castfilm_detection.detection_statistics import CastFilmDetectionStatistics
from detect_core.defect_classifier import DefectClassifier
from detect_core.defect_config import DefectConfig
from detect_core.defect_highlighter import DefectHighlighter
from utils.common_util import modify_ccf_file, write_config_txt, read_latest_pixel, split_image_to_base64, \
    run_async_image_save
from utils.generate_time import get_current_time
from utils.light_util import light_judge

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
READ_TEST_DATA = True
# 进程池初始化器，每个子进程独立持有 Detector
_detector = None


def _init_detector():
    global _detector
    _detector = CastFilmDefectDetector()


def _segment_one_image(image_np):
    """子进程任务：接收灰度图像并执行分割"""
    global _detector
    defects, left_edge_x, right_edge_x = _detector.detect_defects_fast(image_np)
    return defects, left_edge_x, right_edge_x, image_np

class ImageCaptureProcess(Process):
    def __init__(self, stop_event,avg_correction,line_frequency,create_new_folder,film_speed,is_modify_speed,
                 per_pixel,is_modify_pixel, size_queue,image_process_queue, statistical_data,cut_image_queue,classify_queue,sample,user):
        super().__init__()
        self.image_queue = Queue(maxsize=Constants.MAX_BUFFER_IMAGE)
        self.stop_event = stop_event
        self.image_path = Path.test_image_dir

        self.current_frequency = line_frequency
        self.last_frequency = Value('i', Constants.DEFAULT_CCF)
        self.avg_correction = avg_correction
        self.save_dir = None
        self.camera = None
        self.ccf_dir = None
        self.create_new_folder = create_new_folder  # 新增变量
        self.folder_created = False  # 用于跟踪文件夹是否已创建
        self.film_speed = film_speed
        self.is_modify_speed = is_modify_speed
        self.per_pixel = per_pixel
        self.is_modify_pixel = is_modify_pixel
        # mainworker新增参数
        self.size_queue = size_queue
        self.image_process_queue = image_process_queue
        self.statistical_data = statistical_data
        self.cut_image_queue = cut_image_queue
        self.classify_queue = classify_queue
        # 初始化各功能模块
        self.defect_highlighter = DefectHighlighter()
        self.statistics_calculator = CastFilmDetectionStatistics()
        self.detection_statistics = self.statistics_calculator
        self.defect_classifier = DefectClassifier()
        # 标记是否已处理第一张图像
        self.first_image_processed = False
        self.test_image_list = []  # 存储图像列表
        self.current_image_index = 0  # 当前图像索引
        self.sample = sample
        self.user = user
        self.image_save_path = Path.IMAGE_SAVE_DIR
        print(f"已经初始化图像捕获进程 平场校正{self.avg_correction.value}")
    # ./ camera / x64 / Release / camera.dll
    def makedir_ccf(self):
        if os.path.exists(self.ccf_dir):
            logger.info(f"CCF配置文件已存在: {self.ccf_dir}")  # 文件存在
        else:
            modify_ccf_file(os.path.join(Path.CCF_DIR, f"default.ccf"),
                            self.current_frequency.value,
                            Path.CCF_DIR)
            write_config_txt(self.current_frequency.value,self.per_pixel.value,Path.CCF_PIXEL_DIR)
    def init_camera(self):
        print(f"膜速：{self.film_speed.value}，修改；{self.is_modify_speed.value},单位像素：{self.per_pixel.value}，修改：{self.is_modify_pixel.value}")
        self.ccf_dir = os.path.join(Path.CCF_DIR, f"{self.current_frequency.value}.ccf")
        if self.is_modify_pixel.value and self.is_modify_speed.value:
            print(f"由于修改单位像素和拉膜速度，保存新的CCF文件")
            self.makedir_ccf()
        elif self.is_modify_pixel.value:
            print(f"由于修改单位像素，保存新的CCF文件")
            self.makedir_ccf()
        elif self.is_modify_speed.value:
            print(f"由于修改拉膜速度，保存新的CCF文件")
            self.makedir_ccf()
        else:
            self.makedir_ccf()
            print("未修改任何参数，使用默认CCF文件")
        # 新增日志记录采集卡信息
        logger.info(f"正在初始化相机，采集卡型号: Xtium2-CL_MX4_1, CCF路径: {self.ccf_dir}")

        self.camera = DalsaCamera2(Path.CAMERA_DIR)
        # if not self.camera.init("Xtium-CL_MX4_1", "CameraLink Full Mono", "CameraLink_1",
        #                         "Xtium-CL_MX4_1_Serial_0", self.ccf_dir, self.avg_correction.value):
        if self.camera.init("Xtium2-CL_MX4_1", "CameraLink Full Mono", "CameraLink_1", "Xtium2-CL_MX4_1_Serial_0",
                            self.ccf_dir) == False:
            logger.error("初始化相机失败，请检查：")
            logger.error("1. 确认采集卡型号名称是否正确")
            logger.error("2. 确认CCF配置文件路径有效性")
            logger.error("3. 确认Sapera软件是否已正确安装")
            logger.error("4. 检查硬件连接状态")
            exit()
        if not self.camera.lazy_snap() or not self.camera.get_frame():
            logger.error("获取图像失败")
            exit()

    def run(self):
        if not READ_TEST_DATA:
            self.init_camera()
        segmentation_result_queue = Queue(maxsize=Constants.MAX_QUEUE_SIZE)
        # 启动图像采集线程
        capture_thread = threading.Thread(target=self.capture_image,daemon=True)
        capture_thread.start()
        # 启动分类线程
        classification_thread = threading.Thread(
            target=self.classification_worker,
            args=(segmentation_result_queue,),
            daemon=True
        )
        classification_thread.start()
        # 启动多进程分割
        with ProcessPoolExecutor(max_workers=Constants.MAX_WORKERS,
                                 initializer=_init_detector) as segmentation_executor:
            segmentation_futures = []
            while not self.stop_event.is_set():
                #更新定量检测第二次存图逻辑
                self.update_save_directory()
                # 从图像队列中获取图像进行处理
                try:
                    if not self.image_queue.empty():
                        image = self.image_queue.get()
                        self.process_image(image, segmentation_executor, segmentation_futures)
                    # 收集已完成的分割任务
                    self.collect_segmentation_results(segmentation_futures, segmentation_result_queue)
                    time.sleep(0.01)
                except Empty:
                    continue

    def set_row_frequency(self, frequency):
        try:
            if hasattr(frequency, "value"):
                frequency = frequency.value
            frequency = int(frequency)
            self.camera.set_line_rate(frequency)
            print(f"设置行频成功 {frequency}")
        except Exception as e:
            print(f"设置行频失败: {e}，输入类型为 {type(frequency)}")

    def update_save_directory(self):
        """更新保存目录"""
        current_time = get_current_time()
        current_name = (
            f"{current_time}-"
            f"{self.sample}-"
            f"{self.user}"
        )
        if self.create_new_folder.value and not self.folder_created:
            self.save_dir = os.path.join(self.image_save_path,current_name)
            os.makedirs(self.save_dir, exist_ok=True)
            self.folder_created = True
            print(f"创建新文件夹: {self.save_dir}")
            # 重置标志
            self.create_new_folder.value = False
        elif not self.save_dir:
            # 默认文件夹创建逻辑
            self.save_dir = os.path.join(self.image_save_path, current_name)
            os.makedirs(self.save_dir, exist_ok=True)
        self.folder_created = False

    def capture_image(self):
        """采集图像并放入队列"""
        while  not self.stop_event.is_set():
            time.sleep(0.01)  # 添加适当的延时
            if READ_TEST_DATA:
                # 第一次或图像列表为空时重新加载
                if not self.test_image_list:
                    if os.path.exists(self.image_path):
                        self.test_image_list = [f for f in os.listdir(self.image_path)
                                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                        self.current_image_index = 0

                if self.test_image_list:
                    # 循环获取图像
                    image_filename = self.test_image_list[self.current_image_index]
                    image_path = os.path.join(self.image_path, image_filename)
                    time.sleep(1)
                    try:
                        image = Image.open(image_path)
                        logger.debug(f"读取测试图像: {image_path}")
                        # 更新索引，实现循环
                        self.current_image_index = (self.current_image_index + 1) % len(self.test_image_list)
                        self.image_queue.put(image)
                    except Exception as e:
                        logger.error(f"打开图像失败 {image_path}: {e}")
            else:
                # 相机采集逻辑保持不变
                if not self.camera.get_frame() or not self.camera.lazy_snap():
                    logger.error("获取图像失败")
                    return
                if self.current_frequency.value != self.last_frequency.value:
                    self.set_row_frequency(self.current_frequency)
                    self.last_frequency.value = self.current_frequency.value
                image_array = self.camera.frame_buffer_to_image()
                # 异步保存图像（PIL）
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]  # 获取前8位UUID，更短更好看
                filename = f"{timestamp}_{unique_id}.jpg"

                save_path = os.path.join(self.save_dir, filename)
                run_async_image_save(image_array, save_path)

                print(f"图像异步保存中：{save_path}")
                logger.debug(f"图像大小为{image_array.size}")
                self.image_queue.put(image_array)

    def process_image(self, image, segmentation_executor, segmentation_futures):
        """处理单个图像，提交到分割进程池"""
        if isinstance(image, Image.Image):
            image_cv = np.array(image.convert('L'))
        else:
            image_cv = image
        # 设置动态参考灰度值
        if not self.first_image_processed:
            result = light_judge(image_cv)
            if result != Light.NORMAL:
                logger.warning(result)
            det = CastFilmDefectDetector()
            det.set_first_image(image_cv)
            self.first_image_processed = True
        # 记录图像进入队列的时间
        future = segmentation_executor.submit(_segment_one_image, image_cv)
        segmentation_futures.append(future)
        del image


    def collect_segmentation_results(self, segmentation_futures, segmentation_result_queue):
        """收集已完成的分割任务"""
        done_segmentation = [ f for f in segmentation_futures if f.done()]

        for f in done_segmentation:
            try:
                defects, left_edge_x, right_edge_x, image_cv = f.result()
                # 将时间信息传递给下一阶段
                segmentation_result_queue.put((image_cv, defects, left_edge_x, right_edge_x))
            except Exception as e:
                logger.error(f"分割任务出错: {e}")
            segmentation_futures.remove(f)

    def process_classification_batch(self, batch_images, batch_defects, batch_edges):
        """批量执行分类、统计、结果输出"""
        try:
            start_cls = time.time()
            classify_results = self.defect_classifier.classify_defects_batch(batch_images, batch_defects)
            end_cls = time.time()
            logger.info(f"[分类线程] 批量分类完成: {len(batch_images)} 张 | 耗时: {end_cls - start_cls:.3f}s")

            for image_cv, (left_edge_x, right_edge_x), result in zip(batch_images, batch_edges, classify_results):
                defect_infos = result.get('defect_infos', [])
                membrane_bounds = (
                    (int(left_edge_x), int(right_edge_x))
                    if left_edge_x is not None and right_edge_x is not None
                    else None
                )
                self.detection_statistics.update_area_totals(
                    self.statistical_data,
                    image_cv.shape,
                    membrane_bounds,
                )
                statistics_result = self.statistics_calculator.calculate_statistics(defect_infos)

                # 高亮
                pil_image = Image.fromarray(image_cv)
                highlight_image = self.defect_highlighter.highlight_defects(pil_image, statistics_result['boxes'])

                base64_images = split_image_to_base64(highlight_image, 0)  # 垂直切分
                # 输出队列
                self.image_process_queue.put(base64_images)
                self.size_queue.put(statistics_result['size_counts'])
                self.classify_queue.put((statistics_result['class_counts'], statistics_result['class_size_counts']))

                # 裁剪图像
                cut_images = result.get('cut_images', [])
                for i, cut_image in enumerate(cut_images):
                    if i < len(defect_infos):
                        defect_info = defect_infos[i]
                        defect_class = defect_info.get('defect_class', '其它')
                        size_index = defect_info.get('size_index', 0)
                        size_name = DefectConfig.SIZE_LIST[size_index] if 0 <= size_index < len(
                            DefectConfig.SIZE_LIST) else '未知'
                        self.cut_image_queue.put((cut_image, defect_class, size_name))
        except Exception as e:
            logger.error(f"批量分类处理出错: {e}")

    def classification_worker(self, result_queue):
        """分类线程：批量分类 + 统计 + 高亮"""
        batch_images, batch_defects, batch_edges = [], [], []
        while not self.stop_event.is_set():
            try:
                if not result_queue.empty():
                    image_cv, defects, left_edge_x, right_edge_x = result_queue.get_nowait()
                    batch_images.append(image_cv)
                    batch_defects.append(defects)
                    batch_edges.append((left_edge_x, right_edge_x))

                    if len(batch_images) >= Constants.BATCH_SIZE:
                        self.process_classification_batch(batch_images, batch_defects, batch_edges)
                        batch_images, batch_defects, batch_edges = [], [], []
                else:
                    # 处理剩余未满 batch 的情况
                    if batch_images:
                        self.process_classification_batch(batch_images, batch_defects, batch_edges)
                        batch_images, batch_defects, batch_edges = [], [], []
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"分类线程出错: {e}")

