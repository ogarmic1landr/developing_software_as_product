# Profiling Analysis

## Overview

This document records the performance profiling of the vehicle segmentation
pipeline, completed as part of Week 16 of the Developing Software as a Product
course (Task #1: profile the code, Task #2: decide whether to optimize).

The goal was to measure where the pipeline actually spends its time, identify
bottlenecks, and decide whether optimization is worthwhile for this project.

## Methodology

We used Python's built-in `cProfile` module to profile a single end-to-end run
of the pipeline. The profiling script (`scripts/profile_pipeline.py`) loads a
sample aerial traffic image (`images/img_2268.jpg`), instantiates the
`VehicleSegmentationPipeline`, and runs `segment_and_count()` once. The full
output is sorted by cumulative time and the top 20 functions are reported.

To reproduce:

```bash
python -m scripts.profile_pipeline
```

This prints the table to stdout and saves the raw profile data to
`profile_output.prof` for further analysis (e.g., visualization with
`snakeviz` or `tuna`).

## Results

**Total runtime: 0.535 seconds** on a typical input image.

### Time distribution

| Phase | Time | % of total | Description |
|---|---|---|---|
| Model loading (`VehicleSegmentationPipeline.__init__`) | 0.149s | 28% | One-time YOLO weight loading |
| Inference (`segment_and_count`) | 0.358s | 67% | Model prediction + post-processing |
| Image I/O and overhead | ~0.028s | 5% | OpenCV reads, color conversion, etc. |

### Inference breakdown

The 0.358s inference phase decomposes further:

| Function | Time | Notes |
|---|---|---|
| `ultralytics.engine.model.predict` | 0.354s | Ultralytics top-level call |
| `predictor.stream_inference` | 0.239s | Inference loop |
| Neural network forward pass | 0.183s | The actual model computation |
| Convolution operations | 0.144s | Dominant operation in the forward pass |

## Bottleneck Analysis

The dominant cost is the YOLOv8 model itself. Over 99% of runtime occurs
inside third-party library code (`ultralytics` and `torch`). Our own code
in `vehicle_pipeline_draft.py` (the overlay drawing, mask aggregation, and
vehicle counting) contributes a negligible amount of time on top of the
model call.

This is the expected pattern for any computer vision pipeline built around
a deep learning model: the neural network inference dominates wall-clock
time, and the surrounding orchestration code is essentially free in
comparison.

## Decision: Optimization Is Not Warranted

Following Week 16 Task #2 ("decide if optimising the bottlenecks is worth
it"), we conclude that optimization is **not worthwhile** for this project.
The reasoning:

1. **The bottleneck is not in our code.** The only meaningful target is the
   model inference itself, which lives inside `ultralytics` and `torch`. We
   cannot modify these libraries within the scope of a refactoring project.

2. **We are already using the smallest model variant.** The pipeline uses
   `yolov8n-seg.pt` (the "nano" variant), which is the fastest in the
   YOLOv8 segmentation family. Choosing a smaller model would mean
   abandoning YOLOv8 entirely, which is out of scope.

3. **The remaining 33% of time is dominated by one-time setup.** Model
   loading (`__init__`) is amortized across multiple inferences in any
   real-world usage. For batch processing or the Streamlit UI (where the
   pipeline is cached), this cost is paid once.

4. **Sub-second performance is already acceptable.** At 0.535s per image
   end-to-end (including model load) and ~0.36s per inference, the pipeline
   is fast enough for interactive use in the UI and for batch processing
   our 10-image dataset in under 6 seconds.

## What Would Real Optimization Look Like?

For completeness, if performance ever became a constraint, the available
levers would be (in order of impact):

1. **GPU inference.** Moving from CPU to GPU (CUDA or Metal) would reduce
   the forward pass time by roughly 10–100x for this model size.
2. **Batch inference.** Processing multiple images in a single
   `model.predict()` call amortizes Python/PyTorch overhead.
3. **Model quantization or export.** Converting the model to ONNX, OpenVINO,
   or TensorRT format can give 2–5x speedups on CPU.
4. **A smaller or specialized model.** Training a smaller model purpose-built
   for top-down aerial vehicle imagery would likely be both faster and more
   accurate than the general-purpose YOLOv8n.

None of these belong in the scope of this semester's refactoring project,
but they are the right starting points if the tool were to be productionized.

## Reproducibility

- Profiler: `cProfile` (Python standard library)
- Sample input: `images/img_2268.jpg`
- Hardware: CPU only (no GPU acceleration enabled)
- Model: `yolov8n-seg.pt` (default)

Re-running `python -m scripts.profile_pipeline` will produce results within
~10% of the numbers above on similar hardware.
