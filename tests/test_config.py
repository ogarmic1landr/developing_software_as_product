# tests/test_config.py
from src.segmentation_project.config import PipelineConfig


def test_pipeline_config_has_default_model_path():
    cfg = PipelineConfig()
    assert cfg.model_path == "yolov8n-seg.pt"


def test_pipeline_config_default_confidence():
    cfg = PipelineConfig()
    assert cfg.confidence == 0.25


def test_pipeline_config_default_vehicle_class_ids():
    cfg = PipelineConfig()
    # COCO IDs: 2=car, 5=bus, 7=truck
    assert cfg.vehicle_class_ids == {2, 5, 7}


def test_pipeline_config_is_frozen():
    from dataclasses import FrozenInstanceError

    import pytest

    cfg = PipelineConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.confidence = 0.9  # type: ignore[misc]
