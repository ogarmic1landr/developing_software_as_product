"""Profile the vehicle segmentation pipeline on a sample image."""

import cProfile
import pstats
from pathlib import Path

import cv2

from pipeline import VehicleSegmentationPipeline


def run_pipeline():
    """Run the pipeline on one image"""
    image_path = Path("images/img_2268.jpg")
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pipeline = VehicleSegmentationPipeline()
    result = pipeline.segment_and_count(img_rgb)
    return result


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_pipeline()

    profiler.disable()

    # Save raw profile data
    profiler.dump_stats("profile_output.prof")

    # Print top 20 functions by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
