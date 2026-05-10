import cv2
import numpy as np


class Resize:
    @staticmethod
    def apply(image: np.ndarray, size: tuple = (512, 512)) -> np.ndarray:
        """
        Resize image to target size.
        """

        return cv2.resize(image, size)


class RGBConverter:
    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to RGB.
        """

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image


class Grayscale:
    @staticmethod
    def apply(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        """

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image


class Sharpen:
    @staticmethod
    def apply(image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter.
        """

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image


class CLAHE:
    @staticmethod
    def apply(
        image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE enhancement.
        """

        if len(image.shape) != 2:
            raise ValueError("CLAHE expects a grayscale image")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        return clahe.apply(image)
