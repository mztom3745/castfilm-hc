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
            if use_reference and reference_map[cy, cx] >= (reference_threshold + margin):
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
            boxes.append(
                BoundingBox(
                    top=min_row,
                    left=min_col,
                    bottom=max_row,
                    right=max_col,
                    pixels=pixel_count,
                    dark_ratio=ratio,
                )
            )
    return boxes


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
