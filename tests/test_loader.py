import cv2
import numpy as np
import pytest

from src.segmentation_project.data.loader import ImageLoader


def test_list_images_returns_sorted_image_paths(tmp_path):
    # Arrange: create three image files in a temp folder, out of order
    (tmp_path / "b.jpg").write_bytes(b"fake")
    (tmp_path / "a.png").write_bytes(b"fake")
    (tmp_path / "c.jpeg").write_bytes(b"fake")

    # Act
    result = ImageLoader.list_images(tmp_path)

    # Assert: three files, sorted alphabetically
    assert len(result) == 3
    assert [p.name for p in result] == ["a.png", "b.jpg", "c.jpeg"]


def test_list_images_ignores_non_image_files(tmp_path):
    # Arrange: mix of image and non-image files
    (tmp_path / "photo.jpg").write_bytes(b"fake")
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "notes.md").write_text("also not an image")

    # Act
    result = ImageLoader.list_images(tmp_path)

    # Assert: only the image is returned
    assert len(result) == 1
    assert result[0].name == "photo.jpg"


def test_list_images_raises_for_missing_folder(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        ImageLoader.list_images(missing)


def test_list_images_raises_for_empty_folder(tmp_path):
    # tmp_path is empty by default
    with pytest.raises(ValueError):
        ImageLoader.list_images(tmp_path)


def test_load_image_returns_numpy_array(tmp_path):
    # Arrange: create a real image file using OpenCV
    image_path = tmp_path / "real.png"
    fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), fake_image)

    # Act
    loaded = ImageLoader.load_image(image_path)

    # Assert
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (10, 10, 3)


def test_load_image_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError):
        ImageLoader.load_image(missing)
