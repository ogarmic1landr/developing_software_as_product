from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YoloSegmenter:
    @staticmethod
    def load_model(
        model_name: str = "yolov8n-seg.pt",
    ) -> YOLO:
        """
        Load YOLOv8 segmentation model.
        """

        try:
            return YOLO(model_name)

        except Exception as error:
            raise RuntimeError(f"Failed to load YOLO model: {model_name}") from error

    @staticmethod
    def predict(
        model: YOLO,
        image_path: str | Path,
        conf: float = 0.5,
    ):
        """
        Run YOLOv8 segmentation prediction.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = model.predict(
            source=str(image_path),
            conf=conf,
        )

        if not results:
            raise ValueError("YOLO prediction returned no results.")

        return results

    @staticmethod
    def render_result(
        results,
    ) -> np.ndarray:
        """
        Render YOLO segmentation output image.
        """

        if not results:
            raise ValueError("No YOLO results available for rendering.")

        rendered = results[0].plot()

        return rendered
