from __future__ import annotations

import numpy as np


def compute_background_mean(gray: np.ndarray, ratio: float = 0.1) -> float:
    """Compute the average intensity of the left and right background bands."""
    if gray.ndim != 2:
        raise ValueError("Background computation expects a 2D grayscale array.")
    height, width = gray.shape
    if width == 0:
        raise ValueError("Image width must be positive.")
    band_width = max(1, int(width * ratio))
    if band_width * 2 >= width:
        band_width = max(1, width // 4)
    left_band = gray[:, :band_width]
    right_band = gray[:, width - band_width :]
    background_mean = float(np.concatenate((left_band.flatten(), right_band.flatten())).mean())
    return background_mean
