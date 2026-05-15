import numpy as np


class DiceScore:
    @staticmethod
    def apply(
        predicted_mask: np.ndarray,
        ground_truth_mask: np.ndarray,
    ) -> float:
        """
        Compute Dice coefficient between
        predicted and ground truth masks.
        """

        if predicted_mask.shape != ground_truth_mask.shape:
            raise ValueError("Masks must have the same dimensions.")

        predicted_binary = (predicted_mask > 0).astype(np.uint8)

        ground_truth_binary = (ground_truth_mask > 0).astype(np.uint8)

        intersection = np.sum(predicted_binary * ground_truth_binary)

        total = np.sum(predicted_binary) + np.sum(ground_truth_binary)

        if total == 0:
            return 1.0

        dice_score = (2 * intersection) / total

        return float(dice_score)


class IoUScore:
    @staticmethod
    def apply(
        predicted_mask: np.ndarray,
        ground_truth_mask: np.ndarray,
    ) -> float:
        """
        Compute Intersection over Union.
        """

        if predicted_mask.shape != ground_truth_mask.shape:
            raise ValueError("Masks must have the same dimensions.")

        predicted_binary = (predicted_mask > 0).astype(np.uint8)

        ground_truth_binary = (ground_truth_mask > 0).astype(np.uint8)

        intersection = np.sum(predicted_binary * ground_truth_binary)

        union = np.sum((predicted_binary + ground_truth_binary) > 0)

        if union == 0:
            return 1.0

        iou_score = intersection / union

        return float(iou_score)
