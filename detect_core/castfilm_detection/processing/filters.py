from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _ensure_odd(size: int) -> int:
    return size if size % 2 == 1 else size + 1


def box_blur(image: np.ndarray, size: int = 31) -> np.ndarray:
    """Apply a square box blur using an integral image (edge-padded)."""
    if image.ndim != 2:
        raise ValueError("box_blur expects a 2D array.")
    if size <= 1:
        return image.astype(np.float32, copy=True)
    size = _ensure_odd(size)
    radius = size // 2
    padded = np.pad(image.astype(np.float32), ((radius, radius), (radius, radius)), mode="edge")
    integral = padded.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    area = float(size * size)
    top_left = integral[:-size, :-size]
    top_right = integral[:-size, size:]
    bottom_left = integral[size:, :-size]
    bottom_right = integral[size:, size:]
    blurred = (bottom_right - bottom_left - top_right + top_left) / area
    return blurred.astype(np.float32)


def box_blur_pair_shared(image: np.ndarray, sizes: tuple[int, int]) -> dict[int, np.ndarray]:
    """Compute multiple box blurs with a single integral image."""
    if image.ndim != 2:
        raise ValueError("Expected a 2D image.")
    if not sizes:
        return {}
    sizes = tuple(_ensure_odd(size) for size in sizes)
    max_radius = max(size // 2 for size in sizes)
    padded = np.pad(image.astype(np.float32), ((max_radius, max_radius), (max_radius, max_radius)), mode="edge")
    integral = padded.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = image.shape
    results: dict[int, np.ndarray] = {}
    for size in sizes:
        radius = size // 2
        offset = max_radius - radius
        top = offset
        left = offset
        top_left = integral[top : top + height, left : left + width]
        top_right = integral[top : top + height, left + size : left + size + width]
        bottom_left = integral[top + size : top + size + height, left : left + width]
        bottom_right = integral[top + size : top + size + height, left + size : left + size + width]
        area = float(size * size)
        blurred = (bottom_right - bottom_left - top_right + top_left) / area
        results[size] = blurred.astype(np.float32)
    return results


def box_blur_opencv(image: np.ndarray, size: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available; install opencv-python to use blur_mode='opencv'.")
    if size <= 1:
        return image.astype(np.float32, copy=True)
    size = _ensure_odd(size)
    blurred = cv2.blur(image.astype(np.float32), (size, size), borderType=cv2.BORDER_REPLICATE)
    return blurred.astype(np.float32)
