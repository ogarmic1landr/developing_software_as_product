"""Tests for preprocessing image operations."""

import numpy as np

from segmentation_project.preprocessing.preprocessing import (
    CLAHE,
    TARGET_SIZE,
    Grayscale,
    Resize,
    RGBConverter,
    Sharpen,
)

# ---------- Resize ----------


class TestResize:
    def test_resize_to_default_target_size(self):
        image = np.zeros((500, 800, 3), dtype=np.uint8)
        resized = Resize.apply(image)
        # cv2.resize uses (width, height); output shape is (height, width, channels)
        assert resized.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_resize_to_custom_size(self):
        image = np.zeros((500, 800, 3), dtype=np.uint8)
        resized = Resize.apply(image, size=(200, 100))
        assert resized.shape == (100, 200, 3)

    def test_resize_preserves_dtype(self):
        image = np.zeros((500, 800, 3), dtype=np.uint8)
        resized = Resize.apply(image, size=(100, 100))
        assert resized.dtype == np.uint8

    def test_resize_grayscale_image(self):
        image = np.zeros((500, 800), dtype=np.uint8)
        resized = Resize.apply(image, size=(100, 50))
        assert resized.shape == (50, 100)


# ---------- RGBConverter ----------


class TestRGBConverter:
    def test_swaps_blue_and_red_channels(self):
        # BGR pixel [255, 0, 0] (pure blue) should become RGB [0, 0, 255]
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # blue channel in BGR
        rgb = RGBConverter.apply(bgr)
        assert (rgb[:, :, 2] == 255).all()  # red channel in RGB
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == 0).all()

    def test_preserves_shape(self):
        bgr = np.zeros((100, 50, 3), dtype=np.uint8)
        rgb = RGBConverter.apply(bgr)
        assert rgb.shape == bgr.shape

    def test_preserves_dtype(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb = RGBConverter.apply(bgr)
        assert rgb.dtype == np.uint8


# ---------- Grayscale ----------


class TestGrayscale:
    def test_output_is_2d(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = Grayscale.apply(image)
        assert gray.ndim == 2

    def test_output_shape_matches_input_spatial_dims(self):
        image = np.zeros((50, 80, 3), dtype=np.uint8)
        gray = Grayscale.apply(image)
        assert gray.shape == (50, 80)

    def test_all_white_input_produces_all_white_output(self):
        image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        gray = Grayscale.apply(image)
        assert (gray == 255).all()

    def test_all_black_input_produces_all_black_output(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        gray = Grayscale.apply(image)
        assert (gray == 0).all()


# ---------- Sharpen ----------


class TestSharpen:
    def test_preserves_shape(self):
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        sharpened = Sharpen.apply(image)
        assert sharpened.shape == image.shape

    def test_preserves_dtype(self):
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        sharpened = Sharpen.apply(image)
        assert sharpened.dtype == np.uint8

    def test_uniform_image_unchanged(self):
        # Sharpening a constant-value image should leave it (nearly) unchanged
        # because the kernel sums to 1 and there are no edges to enhance.
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        sharpened = Sharpen.apply(image)
        assert np.allclose(sharpened, image)

    def test_works_on_grayscale(self):
        image = np.full((20, 20), 128, dtype=np.uint8)
        sharpened = Sharpen.apply(image)
        assert sharpened.shape == image.shape


# ---------- CLAHE ----------


class TestCLAHE:
    def test_preserves_shape(self):
        image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
        result = CLAHE.apply(image)
        assert result.shape
