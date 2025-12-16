import torch
import json
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from mobilenetv3_small.mobilenetv3_small import mobilenet_v3_small
from .defect_config import DefectConfig
from config.constant import zsz_Constants
from detect_core.castfilm_detection.detection_statistics import CastFilmDetectionStatistics
from loguru import logger
from typing import Optional, Tuple, Union


class DefectClassifier:
    # 默认配置，若 DefectConfig 中提供同名配置将优先使用
    DEFAULT_MODEL_PATH = "mobilenetv3_small/fiber_dot_epoch90.pt"
    DEFAULT_CLASSES_JSON = "mobilenetv3_small/classes_indices_2.json"
    DEFAULT_INPUT_SIZE = 64
    DEFAULT_NUM_CLASSES = 4
    DEFAULT_NORM_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_NORM_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        # 设备选择，优先使用 DefectConfig.DEVICE（可为 'cuda', 'cuda:0', 'cpu' 等），否则自动
        cfg_device = getattr(DefectConfig, 'DEVICE', None)
        if isinstance(cfg_device, str):
            self.device_v3 = torch.device(cfg_device)
        else:
            self.device_v3 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 检查是否使用多GPU
        self.use_multi_gpu = torch.cuda.device_count() > 1 and self.device_v3.type == 'cuda'
        self.num_gpus = torch.cuda.device_count() if self.use_multi_gpu else 1

        # 路径/输入/归一化等配置（去除魔法数字）
        self.model_v3_path = getattr(DefectConfig, 'MODEL_V3_PATH', self.DEFAULT_MODEL_PATH)
        self.model_v3_json = getattr(DefectConfig, 'MODEL_V3_JSON', self.DEFAULT_CLASSES_JSON)
        self.input_size = int(getattr(DefectConfig, 'INPUT_SIZE', self.DEFAULT_INPUT_SIZE))
        self.num_classes = int(getattr(DefectConfig, 'NUM_CLASSES', self.DEFAULT_NUM_CLASSES))
        self.norm_mean = getattr(DefectConfig, 'NORM_MEAN', self.DEFAULT_NORM_MEAN)
        self.norm_std = getattr(DefectConfig, 'NORM_STD', self.DEFAULT_NORM_STD)

        # pre-processing：去掉 Resize，改为自定义等比缩放+居中填充到固定尺寸
        self.transform_v4 = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std)
        ])

        self.model_v3 = None
        self.models_multi_gpu = []  # 用于存储多个GPU上的模型实例
        self.classes_indict = None

        # 初始化统计计算器
        self.stats_calculator = CastFilmDetectionStatistics()
        # base_dark_threshold = getattr(DefectConfig, 'DARK_VALUE', 0.3)
        # self.dark_ratio_threshold = float(getattr(DefectConfig, 'DARK_RATIO_THRESHOLD', base_dark_threshold))
        self.dark_ratio_threshold = zsz_Constants.DARK_RATIO

    def load_model(self):
        """加载分类模型"""
        # 加载类别映射
        try:
            with open(self.model_v3_json, "r", encoding="utf-8") as json_file:
                self.classes_indict = json.load(json_file)
        except Exception as e:
            logger.warning(f"加载类别映射失败: {self.model_v3_json}, {e}")
            self.classes_indict = {}

        if self.use_multi_gpu:
            # 在每个GPU上分别加载模型实例
            for i in range(self.num_gpus):
                device = torch.device(f'cuda:{i}')
                model = mobilenet_v3_small(num_classes=self.num_classes)
                model.to(device)

                try:
                    checkpoint = torch.load(self.model_v3_path, map_location=device)
                except Exception as e:
                    logger.error(f"加载模型权重失败: {self.model_v3_path}, {e}")
                    raise

                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    # 去掉可能的前缀 "model." 或 "module."
                    state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict)
                model.eval()
                self.models_multi_gpu.append(model)

            logger.info(f"分类模型已加载到 {self.num_gpus} 个GPU上")
        else:
            # 构建模型并加载权重
            self.model_v3 = mobilenet_v3_small(num_classes=self.num_classes)
            self.model_v3.to(self.device_v3)

            try:
                checkpoint = torch.load(self.model_v3_path, map_location=self.device_v3)
            except Exception as e:
                logger.error(f"加载模型权重失败: {self.model_v3_path}, {e}")
                raise

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # 去掉可能的前缀 "model." 或 "module."
                state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint

            self.model_v3.load_state_dict(state_dict)
            self.model_v3.eval()
        logger.info(f"分类模型已加载: {self.model_v3_path} @ {self.device_v3}" +
                    (f" (使用 {torch.cuda.device_count()} 个GPU)" if self.use_multi_gpu else ""))

    # 将需要的辅助逻辑提取为私有方法，便于复用与测试
    @staticmethod
    def _to_gray(arr: np.ndarray) -> np.ndarray:
        """将任意 HxW 或 HxWxC 图像转换为灰度图。"""
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)  # type: ignore
        raise ValueError("Unsupported image shape for grayscale conversion")

    @staticmethod
    def _crop_center(img: np.ndarray, cx: float, cy: float, size: int = 64) -> np.ndarray:
        """以 (cx, cy) 为中心在原图上裁剪固定 size x size 的区域；
        若靠边则自动平移窗口保证裁剪区域完整且不做填充。"""
        h, w = img.shape[:2]
        if w >= size and h >= size:
            x1 = max(0, min(w - size, int(round(cx - size / 2))))
            y1 = max(0, min(h - size, int(round(cy - size / 2))))
            x2 = x1 + size
            y2 = y1 + size
            return img[y1:y2, x1:x2] if img.ndim == 2 else img[y1:y2, x1:x2, :]
        # 极端情况：整幅图小于 size，则返回尽可能大的中心区域（可能小于 size）
        x1 = max(0, int(round(cx - size / 2)))
        y1 = max(0, int(round(cy - size / 2)))
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        return img[y1:y2, x1:x2] if img.ndim == 2 else img[y1:y2, x1:x2, :]

    def _map_model_to_defect_class(
        self,
        model_label: str,
        defect_region: np.ndarray,
        dark_ratio: Optional[float] = None,
    ) -> str:
        """将模型粗分类结果映射为业务最终分类（纤维/晶点/黑点/其它）。"""
        if model_label == 'fiber':
            return "纤维"
        if model_label == 'dot':
            if dark_ratio is not None:
                ratio = max(0.0, min(1.0, float(dark_ratio)))
                if ratio >= self.dark_ratio_threshold:
                    return "黑点"
                return "晶点"
            return self.stats_calculator.determine_dot_type(defect_region)
        # 其他模型标签（如 notclass、notclass2）统一归入其它
        return "其它"

    def _predict_multi_gpu(self, img_batch):
        """在多个GPU上进行推理，提高处理效率"""
        # 计算每个GPU处理的数据量
        batch_size = img_batch.size(0)
        num_gpus = len(self.models_multi_gpu)
        chunk_size = batch_size // num_gpus
        remainder = batch_size % num_gpus

        predictions = []

        # 将数据分发到不同GPU上进行推理
        start_idx = 0
        for i, model in enumerate(self.models_multi_gpu):
            # 计算当前GPU处理的数据量（前面的GPU多处理一个样本，如果不能整除）
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size

            if start_idx >= batch_size:
                break

            # 获取当前GPU需要处理的数据
            chunk_data = img_batch[start_idx:end_idx]

            # 在对应的GPU上进行推理
            device = torch.device(f'cuda:{i}')
            chunk_data = chunk_data.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(chunk_data)
                chunk_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.append(chunk_predictions)

            start_idx = end_idx

        # 合并所有GPU的预测结果
        if predictions:
            return np.concatenate(predictions)
        else:
            return np.array([])

    def classify_defects(self, image, defects):
        """
        处理检测到的缺陷并进行分类
        :param image: 原始图像 (numpy.ndarray)
        :param defects: 检测到的缺陷列表 [(x, y, w, h, area[, dark_ratio]), ...]
        :return: dict，包含：
                 {
                   'defect_infos': 缺陷信息列表,
                   'cut_images': 裁剪的缺陷图像列表
                 }
        """
        # 处理每个检测到的缺陷
        defect_images = []
        defect_infos = []

        H, W = image.shape[:2]
        for defect in defects:
            if len(defect) >= 6:
                x, y, w, h, area, dark_ratio = defect[:6]
            else:
                x, y, w, h, area = defect
                dark_ratio = None
            # 坐标裁剪，避免越界或负索引
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(W, int(x + w))
            y2 = min(H, int(y + h))
            if x1 >= x2 or y1 >= y2:
                continue

            # 提取缺陷区域（用于后续黑点/晶点判定的统计）
            if image.ndim == 2:
                defect_region = image[y1:y2, x1:x2]
            else:
                defect_region = image[y1:y2, x1:x2, :]

            # 在原图上以缺陷中心裁剪固定 64x64（不填充）
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cut_image = self._crop_center(image, cx, cy, self.input_size)
            # 确保裁剪结果为固定尺寸；若原图过小导致无法裁得 64x64，则跳过该缺陷
            if cut_image.shape[0] != self.input_size or cut_image.shape[1] != self.input_size:
                logger.debug(f"跳过尺寸不足的裁剪图: {cut_image.shape}")
                continue

            # 根据面积确定尺寸范围
            size_index = self.stats_calculator.calculate_size_index(area)

            # 准备用于分类的图像数据
            defect_images.append(cut_image)
            defect_infos.append({
                'size_index': size_index,
                'defect_region': defect_region,
                'pixel_area': area,
                'coord': (x1, y1, x2 - x1, y2 - y1),
                'dark_ratio': float(dark_ratio) if dark_ratio is not None else 0.0,
            })

        # 分类缺陷
        if defect_images:
            # 加载模型（如果尚未加载）
            if self.model_v3 is None and not self.models_multi_gpu:
                self.load_model()

            # 预处理图像：直接对裁剪得到的 64x64 图做灰度+标准化
            tensors = []
            for im in defect_images:
                if isinstance(im, np.ndarray):
                    if im.ndim == 2:
                        pil_im = Image.fromarray(im, mode='L')
                    elif im.ndim == 3:
                        # OpenCV 常为 BGR，转为 RGB 再交由 torchvision
                        pil_im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # type: ignore
                    else:
                        raise ValueError("Unsupported image shape for transform")
                else:
                    pil_im = im
                tensors.append(self.transform_v4(pil_im))

            # 批量预测
            img_batch = torch.stack(tensors, dim=0)

            # 模型非空保护，便于静态检查与运行时安全
            if self.model_v3 is None and not self.models_multi_gpu:
                raise RuntimeError("分类模型尚未加载")

            # 根据是否使用多GPU选择推理方式
            if self.use_multi_gpu and self.models_multi_gpu:
                # 使用多GPU推理 - 将数据分批在不同GPU上处理
                predictions = self._predict_multi_gpu(img_batch)
            else:
                # 单GPU/CPU推理
                self.model_v3.eval()
                with torch.no_grad():
                    outputs = self.model_v3(img_batch.to(self.device_v3))
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # 映射为类别名称并进行最终分类
            for i, p in enumerate(predictions):
                if self.classes_indict is not None:
                    model_result = self.classes_indict.get(str(int(p)), "其它")
                else:
                    model_result = "其它"  # 默认类别

                # 统一映射逻辑
                defect_class = self._map_model_to_defect_class(
                    model_result,
                    defect_infos[i]['defect_region'],
                    defect_infos[i].get('dark_ratio'),
                )

                # 更新缺陷信息，添加分类结果
                defect_infos[i]['defect_class'] = defect_class
        else:
            logger.debug("没有检测到缺陷")

        return {
            'defect_infos': defect_infos,
            'cut_images': defect_images
        }
    def classify_defects_batch(self, images_list, defects_list):
        """
        批量处理多张图像的缺陷分类
        :param images_list: [img1, img2, ...] 每张为灰度或彩色 numpy.ndarray
        :param defects_list: [defects_img1, defects_img2, ...] 每个元素是对应图像的缺陷列表
        :return: list of classify_result（和单张调用返回格式相同），按输入顺序
        """
        results = []
        for image, defects in zip(images_list, defects_list):
            results.append(self.classify_defects(image, defects))
        return results
