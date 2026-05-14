import cv2
import numpy as np

class BinarizeMask:
    @staticmethod
    def apply(
        mask: np.ndarray,
        threshold: int = 127,
    ) -> np.ndarray:
        """
        Convert manual mask into a binary mask.
        """

        _, binary_mask = cv2.threshold(
            mask,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )

        return binary_mask



class Overlay:
    @staticmethod
    def apply(
        grayscale_image: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Overlay binary mask onto grayscale image.
        """

        if grayscale_image.shape != binary_mask.shape:
            raise ValueError(
                "Grayscale image and binary mask "
                "must have the same dimensions."
            )

        overlay = grayscale_image.copy()

        overlay[binary_mask == 255] = 255

        return overlay