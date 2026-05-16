import numpy as np
import pytest

from segmentation_project.model_evaluation.evaluation import DiceScore, IoUScore


def test_dice_identical_masks():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert DiceScore.apply(mask, mask) == 1.0


def test_dice_disjoint_masks():
    a = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    b = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    assert DiceScore.apply(a, b) == 0.0


def test_dice_shape_mismatch_raises():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        DiceScore.apply(a, b)


def test_iou_identical_masks():
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    assert IoUScore.apply(mask, mask) == 1.0
