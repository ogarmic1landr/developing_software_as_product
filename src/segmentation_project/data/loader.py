from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    def __init__(
        self,
        image_folder: str = "images",
        mask_folder: str = "masks_manual",
        marker: str = "pyproject.toml",
    ):
        self.project_root = self.find_project_root(marker)
        self.image_dir = self.project_root / image_folder
        self.mask_dir = self.project_root / mask_folder

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

    def get_mask_path(
        self,
        image_path: str | Path,
        ext: tuple = (".jpg", ".png", ".jpeg", ".tif"),
    ) -> Path:
        """
        Match image filename to corresponding manual mask.
        """

        image_path = Path(image_path)

        image_stem = image_path.stem

        for extension in ext:
            mask_name = f"{image_stem}_mask_final{extension}"

            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                return mask_path

        raise FileNotFoundError(f"No matching mask found for: {image_stem}")

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

    @staticmethod
    def load_mask(path: str | Path) -> np.ndarray:
        """
        Load manual mask as grayscale image.
        """

        path = Path(path)

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {path}")

        return mask
