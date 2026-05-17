"""Tests for ground truth mask operations."""

import numpy as np
import pytest

from segmentation_project.ground_truth.overlay import BinarizeMask, Overlay

# ---------- BinarizeMask ----------


class TestBinarizeMask:
    def test_output_contains_only_zero_and_255(self):
        mask = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        unique_values = set(np.unique(result).tolist())
        assert unique_values.issubset({0, 255})

    def test_values_above_default_threshold_become_255(self):
        # Default threshold is 127; cv2.THRESH_BINARY uses strict greater-than
        mask = np.array([[128, 200, 255]], dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        assert (result == 255).all()

    def test_values_at_or_below_default_threshold_become_zero(self):
        mask = np.array([[0, 50, 127]], dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        assert (result == 0).all()

    def test_custom_threshold(self):
        mask = np.array([[50, 100, 150, 200]], dtype=np.uint8)
        result = BinarizeMask.apply(mask, threshold=99)
        # cv2.THRESH_BINARY: values > threshold become 255, others become 0
        expected = np.array([[0, 255, 255, 255]], dtype=np.uint8)
        assert np.array_equal(result, expected)

    def test_preserves_shape(self):
        mask = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        assert result.shape == mask.shape

    def test_preserves_dtype(self):
        mask = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        assert result.dtype == np.uint8

    def test_all_zero_input_returns_all_zero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        assert (result == 0).all()

    def test_all_255_input_returns_all_255(self):
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = BinarizeMask.apply(mask)
        assert (result == 255).all()


# ---------- Overlay ----------


class TestOverlay:
    def test_preserves_shape(self):
        gray = np.full((20, 30), 100, dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=np.uint8)
        result = Overlay.apply(gray, mask)
        assert result.shape == gray.shape

    def test_preserves_dtype(self):
        gray = np.full((20, 30), 100, dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=np.uint8)
        result = Overlay.apply(gray, mask)
        assert result.dtype == np.uint8

    def test_mask_pixels_become_white(self):
        gray = np.full((4, 4), 100, dtype=np.uint8)
        mask = np.array(
            [[0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
        result = Overlay.apply(gray, mask)
        # Top-right 2x2 should be 255, rest should remain 100
        assert (result[:2, 2:] == 255).all()
        assert (result[:2, :2] == 100).all()
        assert (result[2:, :] == 100).all()

    def test_does_not_modify_original_image(self):
        gray = np.full((4, 4), 100, dtype=np.uint8)
        mask = np.full((4, 4), 255, dtype=np.uint8)
        original = gray.copy()
        _ = Overlay.apply(gray, mask)
        assert np.array_equal(gray, original)

    def test_empty_mask_returns_unchanged_image(self):
        gray = np.full((10, 10), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = Overlay.apply(gray, mask)
        assert np.array_equal(result, gray)

    def test_full_mask_returns_all_white(self):
        gray = np.full((10, 10), 100, dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = Overlay.apply(gray, mask)
        assert (result == 255).all()

    def test_shape_mismatch_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        mask = np.zeros((10, 20), dtype=np.uint8)
        with pytest.raises(ValueError, match="same dimensions"):
            Overlay.apply(gray, mask)

    def test_mask_must_have_value_255_to_overlay(self):
        # The implementation specifically checks `binary_mask == 255`,
        # so values like 1 or 127 should NOT overlay.
        gray = np.full((4, 4), 100, dtype=np.uint8)
        mask = np.full((4, 4), 127, dtype=np.uint8)
        result = Overlay.apply(gray, mask)
        assert (result == 100).all()


# ---------- Composition ----------


def test_binarize_then_overlay_pipeline():
    """Typical usage: raw mask -> binarize -> overlay on grayscale image."""
    gray = np.full((6, 6), 80, dtype=np.uint8)
    raw_mask = np.array(
        [
            [0, 0, 0, 200, 200, 200],
            [0, 0, 0, 200, 200, 200],
            [0, 0, 0, 200, 200, 200],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    binary = BinarizeMask.apply(raw_mask)
    overlay = Overlay.apply(gray, binary)

    # Top-right 3x3 region should be white (255), rest should remain 80
    assert (overlay[:3, 3:] == 255).all()
    assert (overlay[:3, :3] == 80).all()
    assert (overlay[3:, :] == 80).all()
