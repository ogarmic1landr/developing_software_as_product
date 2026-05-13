import numpy as np
import pytest

from src.segmentation_project.preprocessing import (
    CLAHE,
    Grayscale,
    Resize,
    RGBConverter,
    Sharpen,
)


def test_resize_changes_image_to_target_size():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = Resize.apply(image, size=(64, 64))
    assert resized.shape == (64, 64, 3)


def test_rgb_converter_swaps_red_and_blue_channels():
    # BGR pixel: blue=10, green=20, red=30
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb = RGBConverter.bgr_to_rgb(bgr)
    # After conversion the same pixel should read R=30, G=20, B=10
    assert tuple(rgb[0, 0]) == (30, 20, 10)


def test_grayscale_returns_2d_array():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    gray = Grayscale.apply(image)
    assert gray.ndim == 2
    assert gray.shape == (50, 50)


def test_sharpen_preserves_image_shape():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    sharpened = Sharpen.apply(image)
    assert sharpened.shape == image.shape


def test_clahe_raises_on_color_image():
    color_image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        CLAHE.apply(color_image)


def test_clahe_returns_same_shape_for_grayscale():
    gray_image = np.zeros((50, 50), dtype=np.uint8)
    result = CLAHE.apply(gray_image)
    assert result.shape == (50, 50)
