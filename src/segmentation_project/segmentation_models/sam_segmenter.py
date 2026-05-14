from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
)

from segmentation_project.preprocessing.preprocessing import (
    Resize,
    RGBConverter,
)


class SAMSegmenter:
    @staticmethod
    def download_checkpoint(
        checkpoint_path: str | Path = "sam_vit_b_01ec64.pth",
        url: str = ("https://dl.fbaipublicfiles.com/" "segment_anything/sam_vit_b_01ec64.pth"),
    ) -> Path:
        """
        Download SAM checkpoint if it does not exist.
        """

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"Downloading SAM checkpoint " f"to {checkpoint_path}...")

            urlretrieve(url, checkpoint_path)

            print("SAM checkpoint downloaded " "successfully.")

        return checkpoint_path

    @staticmethod
    def load_model(
        checkpoint_path: str | Path = "sam_vit_b_01ec64.pth",
        model_type: str = "vit_b",
        device: str | None = None,
    ):
        """
        Load SAM model.
        """

        checkpoint_path = SAMSegmenter.download_checkpoint(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))

        sam.to(device=device)

        return sam

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize and convert image to RGB.
        """

        resized = Resize.apply(image, size=(512, 512))

        rgb_image = RGBConverter.apply(resized)

        return rgb_image

    @staticmethod
    def generate_masks(
        sam,
        image: np.ndarray,
    ) -> list:
        """
        Generate SAM masks for an image.
        """

        mask_generator = SamAutomaticMaskGenerator(sam)

        masks = mask_generator.generate(image)

        if not masks:
            raise ValueError("SAM returned no masks.")

        return masks

    @staticmethod
    def filter_small_masks(
        masks: list,
        max_area: int = 250,
    ) -> list:
        """
        Filter masks by maximum area.
        """

        filtered_masks = []

        for mask in masks:
            segmentation = mask["segmentation"]

            area = np.sum(segmentation)

            if area <= max_area:
                filtered_masks.append(mask)

        return filtered_masks

    @staticmethod
    def create_binary_mask(
        image_shape: tuple,
        masks: list,
    ) -> np.ndarray:
        """
        Create white-on-black binary mask.
        """

        height, width = image_shape[:2]

        binary_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        for mask in masks:
            segmentation = mask["segmentation"]

            binary_mask[segmentation] = 255

        return binary_mask
