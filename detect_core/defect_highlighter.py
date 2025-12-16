from PIL import ImageDraw
from .defect_config import DefectConfig


class DefectHighlighter:
    def __init__(self):
        pass

    def highlight_defects(self, image, total_defect_cord):
        """
        将缺陷的最小包围矩形内所有像素高亮。
        :param image: PIL.Image 对象
        :param total_defect_cord: 缺陷坐标列表 [(x, y, w, h), ...]
        :return: 处理后的图像对象
        """
        # 若为灰度或非 RGB 模式，转换为 RGB 以支持 (R,G,B) 颜色
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image)

        use_fill = DefectConfig.HIGHLIGHT_USE_FILL
        outline = DefectConfig.HIGHLIGHT_OUTLINE_COLOR
        fill = DefectConfig.HIGHLIGHT_FILL_COLOR if use_fill else None

        # 遍历所有缺陷区域并绘制矩形
        for x, y, w, h in total_defect_cord:
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)

        return image