from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    def __init__(
        self,
        image_folder: str = "images",
        marker: str = "pyproject.toml",
    ):
        self.project_root = self.find_project_root(marker)
        self.image_dir = self.project_root / image_folder

    @staticmethod
    def find_project_root(marker="pyproject.toml") -> Path:
        """
        Find project root by locating pyproject.toml.
        """

        path = Path.cwd()

        while path != path.parent:
            if (path / marker).exists():
                return path

            path = path.parent

        raise FileNotFoundError(f"Project root not found (missing {marker})")

    def list_images(
        self,
        ext: tuple = (".jpg", ".png", ".jpeg", ".tif"),
    ) -> list[Path]:
        """
        List image files from the configured image directory.
        """

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {self.image_dir}")

        image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.is_file() and f.suffix.lower() in ext]
        )

        if not image_files:
            raise ValueError(f"No image files found in {self.image_dir}")

        return image_files

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """
        Load image from path.
        """

        path = Path(path)

        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        return image
