from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .config import PipelineConfig


class VehicleSegmentationPipeline:
    """Loads a YOLOv8 segmentation model and provides segmentation + counting."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        vehicle_class_ids: Iterable[int] | None = None,
    ) -> None:
        cfg = PipelineConfig()
        self.model_path = model_path or cfg.model_path
        self.confidence = cfg.confidence if confidence is None else confidence
        self.vehicle_class_ids = set(vehicle_class_ids or cfg.vehicle_class_ids)
        self.model = YOLO(self.model_path)

    def segment_and_count(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray | int | List[Tuple[int, int, int, int]]]:
        """Run model inference, create visual overlays, and count vehicle instances."""
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Input image is empty.")

        results = self.model.predict(source=image_rgb, conf=self.confidence, verbose=False)
        result = results[0]

        overlay = image_rgb.copy()
        mask_layer = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        boxes_out: List[Tuple[int, int, int, int]] = []

        if result.boxes is None or result.boxes.cls is None:
            return {
                "overlay_rgb": overlay,
                "binary_mask": mask_layer,
                "vehicle_count": 0,
                "boxes": boxes_out,
            }

        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)

        mask_data = None
        if result.masks is not None and result.masks.data is not None:
            mask_data = result.masks.data.cpu().numpy().astype(np.uint8)

        vehicle_count = 0
        for idx, cls_id in enumerate(classes):
            if cls_id not in self.vehicle_class_ids:
                continue

            vehicle_count += 1
            x1, y1, x2, y2 = xyxy[idx]
            boxes_out.append((x1, y1, x2, y2))

            if mask_data is not None and idx < len(mask_data):
                mask = mask_data[idx] > 0
                mask_layer[mask] = 255
                overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

            label = f"vehicle {confidences[idx]:.2f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return {
            "overlay_rgb": overlay,
            "binary_mask": mask_layer,
            "vehicle_count": vehicle_count,
            "boxes": boxes_out,
        }
