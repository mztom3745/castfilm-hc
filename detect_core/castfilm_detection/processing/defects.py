from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BoundingBox:
    top: int
    left: int
    bottom: int
    right: int
    pixels: int
    dark_pixels:int
    dark_ratio: float = 0.0
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

from config.constant import zsz_Constants
def extract_bounding_boxes(
    mask: np.ndarray,
    min_pixels: int = zsz_Constants.MIN_PIXELS,
    reference_map: Optional[np.ndarray] = None,
    reference_threshold: Optional[float] = None,
    reference_margin: float = 0.0,
    #这就是一个手写的 DFS（深度优先搜索）：
    # 从当前起点 (y, x) 出发
    # 一路把与它 8 邻相连的True 像素都搜出来
    # 这些像素 = 一个连通缺陷区域
) -> List[BoundingBox]:
    """Return bounding boxes for connected components inside the mask."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    boxes: List[BoundingBox] = []
    coords = np.argwhere(binary)
    if coords.size == 0:
        return boxes
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    use_reference = reference_map is not None and reference_threshold is not None
    if use_reference:
        if reference_map.shape != binary.shape:
            raise ValueError("reference_map must match mask shape.")
        margin = float(reference_margin)
    else:
        margin = 0.0
    for y, x in coords:
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        min_row = max_row = y
        min_col = max_col = x
        pixel_count = 0
        dark_pixels = 0
        while stack:
            cy, cx = stack.pop()
            pixel_count += 1
            if use_reference and reference_map[cy, cx] >= margin:
                dark_pixels += 1
            if cy < min_row:
                min_row = cy
            if cy > max_row:
                max_row = cy
            if cx < min_col:
                min_col = cx
            if cx > max_col:
                max_col = cx
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if pixel_count >= min_pixels:
            ratio = (dark_pixels / pixel_count) if (use_reference and pixel_count) else 0.0
            # if _is_equivalent_diameter_between_25_50_um(pixel_count):
            #     if _is_max_diameter_between_30_45_um(
            #         min_row=min_row,
            #         max_row=max_row,
            #         min_col=min_col,
            #         max_col=max_col,
            #         um_per_pixel=DefectConfig.UM_PER_PIXEL,
            #     ):
            #         boxes.append(
            #             BoundingBox(
            #                 top=min_row,
            #                 left=min_col,
            #                 bottom=max_row,
            #                 right=max_col,
            #                 pixels=pixel_count,
            #                 dark_pixels = dark_pixels,
            #                 dark_ratio=ratio,
            #             )
            #         )
            # else:
            ## 

            if use_reference and _is_equivalent_diameter_between_25_50_um(pixel_count):
                # 1) 取 ROI（原框范围）
                y0, y1 = min_row, max_row
                x0, x1 = min_col, max_col

                roi_binary = binary[y0 : y1 + 1, x0 : x1 + 1]
                roi_ref = reference_map[y0 : y1 + 1, x0 : x1 + 1]

                # 2) 更高阈值生成 refine mask：>= GRAY_VALUE_50 认为是缺陷
                refine = (roi_ref >= float(zsz_Constants.GRAY_VALUE_50))

                # 3) 和原始候选缺陷 mask 取交集（很重要，防止把背景噪声引入）
                refine &= roi_binary

                # 4) 在 ROI 内做 DFS，可能得到多个新框，也可能为空
                refined_components = _dfs_boxes_in_binary(refine, min_pixels=min_pixels)

                if refined_components:
                    for r_top, r_left, r_bottom, r_right, r_pixels in refined_components:
                        # 5) 把 ROI 坐标转回全图坐标
                        top = y0 + r_top
                        left = x0 + r_left
                        bottom = y0 + r_bottom
                        right = x0 + r_right

                        # 重新计算 dark_pixels / ratio（基于 reference_map & threshold）
                        # 这里按照 “>= GRAY_VALUE_50” 的像素数作为 dark_pixels（定义更一致）
                        roi_ref2 = reference_map[top : bottom + 1, left : right + 1]
                        # roi_bin2 = binary[top : bottom + 1, left : right + 1]
                        # 注意：只统计仍在 binary 内的像素
                        dark_pixels2 = int(np.sum(roi_ref2 >= float(zsz_Constants.DARK_VALUE_50)))
                        # pixels2 = int(np.sum(roi_bin2))
                        ratio2 = (dark_pixels2 / r_pixels) if r_pixels > 0 else 0.0

                        boxes.append(
                            BoundingBox(
                                top=top,
                                left=left,
                                bottom=bottom,
                                right=right,
                                pixels=r_pixels,          # 或者用 r_pixels（refine 连通域像素数）
                                dark_pixels=dark_pixels2,
                                dark_ratio=ratio2,
                            )
                        )
                # refined_components 为空：表示该候选框在更高阈值下“消失”，那就不 append（即被过滤掉）
            else:
                # ✅ 非 25~50um 或没 reference_map：走原逻辑
                boxes.append(
                    BoundingBox(
                        top=min_row,
                        left=min_col,
                        bottom=max_row,
                        right=max_col,
                        pixels=pixel_count,
                        dark_pixels=dark_pixels,
                        dark_ratio=ratio,
                    )
                )
    return boxes

import math
from detect_core.defect_config import DefectConfig



def _dfs_boxes_in_binary(
    binary: np.ndarray,
    min_pixels: int,
) -> List[Tuple[int, int, int, int, int]]:
    """
    对一个 bool mask(binary) 做 8 邻域 DFS，返回若干连通域的框：
    (top, left, bottom, right, pixels)
    坐标是 binary 的局部坐标（从 0 开始）。
    """
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    coords = np.argwhere(binary)
    if coords.size == 0:
        return []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    out = []

    for y, x in coords:
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True

        min_row = max_row = y
        min_col = max_col = x
        pixel_count = 0

        while stack:
            cy, cx = stack.pop()
            pixel_count += 1
            if cy < min_row: min_row = cy
            if cy > max_row: max_row = cy
            if cx < min_col: min_col = cx
            if cx > max_col: max_col = cx

            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        if pixel_count >= min_pixels:
            out.append((min_row, min_col, max_row, max_col, pixel_count))

    return out

def _is_equivalent_diameter_between_25_50_um(
    pixel_count: int,
    um_per_pixel: float = DefectConfig.UM_PER_PIXEL,
) -> bool:
    """
    使用缺陷像素面积，计算等效圆直径（Equivalent Circle Diameter, ECD），
    判断是否在 [25, 50) µm

    语义：Size compute mode = Area（以直径形式表达）
    """
    if pixel_count <= 0:
        return False

    # 像素面积 -> 物理面积 (µm²)
    area_um2 = pixel_count * (um_per_pixel ** 2)

    # 等效圆直径 (µm)
    diameter_um = 2.0 * math.sqrt(area_um2 / math.pi)

    return 25.0 <= diameter_um < 50.0


def _is_max_diameter_between_30_45_um(
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
    um_per_pixel = DefectConfig.UM_PER_PIXEL,
) -> bool:
    """
    判断缺陷的最大直径（bounding box 最大边）是否在 [25, 50) µm
    对应 Size compute mode = Diameter
    """
    # bounding box 尺寸（像素）
    height_px = max_row - min_row + 1
    width_px = max_col - min_col + 1

    # 最大直径（像素）
    max_diameter_px = max(height_px, width_px)

    # 转换为微米
    max_diameter_um = max_diameter_px * um_per_pixel

    return 30.0 <= max_diameter_um < 45.0

def suppress_dense_clusters(
    mask: np.ndarray,
    tile_size: int = 128,
    density_threshold: float = 0.03,
    #256×256 tile 中如果有超过 8% 的像素被判成缺陷，就视为假缺陷区域。
) -> np.ndarray:
    """Remove tile regions where defect density is unreasonably high."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if density_threshold <= 0:
        return mask
    binary = np.asarray(mask, dtype=bool).copy()
    height, width = binary.shape
    if not binary.any():
        return binary
    for top in range(0, height, tile_size):
        bottom = min(height, top + tile_size)
        for left in range(0, width, tile_size):
            right = min(width, left + tile_size)
            block = binary[top:bottom, left:right]
            if block.size == 0:
                continue
            density = block.mean()
            if density >= density_threshold:
                block[:] = False
    return binary
