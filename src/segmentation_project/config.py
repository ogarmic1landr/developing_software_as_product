from dataclasses import dataclass, field
from typing import Set


@dataclass(frozen=True)
class PipelineConfig:
    """Default configuration for vehicle segmentation and counting."""

    model_path: str = "yolov8n-seg.pt"
    confidence: float = 0.25
    vehicle_class_ids: Set[int] = field(default_factory=lambda: {2, 5, 7})
