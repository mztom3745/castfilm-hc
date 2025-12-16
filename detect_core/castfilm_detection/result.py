from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from detect_core.castfilm_detection.processing.defects import BoundingBox


@dataclass
class DetectionResult:
    image_name: str
    background_mean: float
    threshold_value: float | None = None
    boxes: List[BoundingBox] | None = None
    size_distribution: Dict[str, int] | None = None
    membrane_bounds: Tuple[int, int] | None = None
    out_of_range: bool = False
    suggestion: str | None = None
    defect_count: int = 0
    processing_time: float = 0.0
    timings: Dict[str, float] | None = None
    membrane_source: str = "detected"
    membrane_issue: bool = False
    membrane_issue_reason: str | None = None
    detection_anomaly: bool = False
    detection_anomaly_reason: str | None = None
    detection_area: Dict[str, float | int] | None = None
    dot_class_counts: Dict[str, int] | None = None
