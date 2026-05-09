import cv2
import numpy as np
from pathlib import Path


class ImageLoader:

    @staticmethod
    def list_images(
        folder: str | Path,
        ext: tuple = (".jpg", ".png", ".jpeg", ".tif")
    ) -> list[Path]:
        
        """
        List image files from a directory.
        """

        folder_path = Path(folder)


        if not folder_path.is_dir():
            raise FileNotFoundError(
                f"Directory does not exist or is invalid: {folder_path}"
            )

        image_files = sorted([
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in ext
        ])
        

        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")

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