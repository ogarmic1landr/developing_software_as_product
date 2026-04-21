# GitHub Issues Roadmap (Chronological)

This roadmap is ordered so the repository evolves from a large notebook into a clean,
modular, and maintainable software product.

## Phase 1: Foundation and Structure

### Issue 1: Create package structure for modular code
- Goal: Move reusable code out of the notebook into a Python package.
- Tasks:
  - Create `src/segmentation_project/`.
  - Add `__init__.py` and base module files.
  - Add clear folder conventions for `images`, `masks_manual`, `masks_auto`, `runs`.
- Acceptance criteria:
  - Package imports work from notebook and scripts.
  - No hardcoded absolute machine paths in modules.

### Issue 2: Centralize config and project paths
- Goal: Replace hardcoded constants and scattered paths.
- Tasks:
  - Add `config.py` with dataclass-based settings.
  - Define model paths, image size, threshold defaults, and directory config.
- Acceptance criteria:
  - Notebook and scripts read config values from one place.

## Phase 2: Refactor Core Logic

### Issue 3: Extract image I/O and path utilities
- Goal: Reuse loading/saving logic and reduce repeated notebook code.
- Tasks:
  - Add `io_utils.py` with `list_images`, `load_image`, `save_image`, `ensure_dir`, `find_manual_mask`.
- Acceptance criteria:
  - No repeated image loading boilerplate in notebook cells.

### Issue 4: Extract preprocessing module
- Goal: Make preprocessing explicit, testable, and reusable.
- Tasks:
  - Add `preprocessing.py` with resize, grayscale, unsharp masking, CLAHE.
  - Add single-image and batch preprocess functions.
- Acceptance criteria:
  - Preprocessing steps are callable with one function per step.

### Issue 5: Add manual mask validation module
- Goal: Standardize validation against manual masks.
- Tasks:
  - Add `manual_masking.py` with binarization, alignment check, overlay function.
- Acceptance criteria:
  - Manual mask checks can run on one image or full dataset with same API.

### Issue 6: Add SAM wrapper class
- Goal: Isolate model loading and mask generation state.
- Tasks:
  - Add `sam_segmenter.py` with `SamSegmenter` class.
  - Include conversion utilities (`masks_to_binary`, area filtering).
- Acceptance criteria:
  - SAM is initialized once and reused.

### Issue 7: Add YOLO wrapper class
- Goal: Keep YOLO logic separate and maintainable.
- Tasks:
  - Add `yolo_segmenter.py` with `YoloSegmenter` class.
  - Provide inference methods and optional prediction image retrieval.
- Acceptance criteria:
  - Notebook no longer contains direct low-level YOLO setup code.

### Issue 8: Add evaluation module with metrics
- Goal: Consolidate all evaluation metrics.
- Tasks:
  - Add `evaluation.py` with Dice, IoU, Precision, Recall, dataset evaluator.
- Acceptance criteria:
  - One canonical Dice implementation is used everywhere.

### Issue 9: Add visualization helpers module
- Goal: Remove repeated plotting code.
- Tasks:
  - Add `visualization.py` with side-by-side, grid, overlay, and metric-plot functions.
- Acceptance criteria:
  - Notebook plots call helper functions instead of duplicating Matplotlib blocks.

## Phase 3: Notebook Cleanup

### Issue 10: Rewrite notebook to orchestration-only style
- Goal: Keep notebook readable and focused on narrative/results.
- Tasks:
  - Keep markdown explanations.
  - Replace long cells with short calls into modules.
  - Remove duplicate imports and duplicate loops.
- Acceptance criteria:
  - Notebook cells are short and understandable.
  - Logic is primarily in `.py` files.

### Issue 11: Split monolithic notebook into 4 focused notebooks
- Goal: Improve clarity and debugging.
- Tasks:
  - Create:
    - `01_data_and_preprocessing.ipynb`
    - `02_manual_mask_validation.ipynb`
    - `03_auto_segmentation_experiments.ipynb`
    - `04_evaluation_and_report.ipynb`
- Acceptance criteria:
  - Each notebook has one clear objective and can run independently.

## Phase 4: Quality and Reproducibility

### Issue 12: Add tests for core logic
- Goal: Prevent regressions during further refactor.
- Tasks:
  - Add `tests/test_preprocessing.py`.
  - Add `tests/test_evaluation.py`.
  - Add `tests/test_mask_filters.py`.
- Acceptance criteria:
  - Core functions have automated tests and pass in CI.

### Issue 13: Add linting and formatting tools
- Goal: Keep code style consistent across collaborators.
- Tasks:
  - Configure `black`, `ruff`, and optional `mypy`.
  - Add pre-commit hooks.
- Acceptance criteria:
  - Formatting and lint checks run locally and in CI.

### Issue 14: Improve requirements and environment strategy
- Goal: Make setup reliable for all contributors.
- Tasks:
  - Keep conda environment file.
  - Add a concise pip requirements file for runtime app usage.
  - Document environment setup in README.
- Acceptance criteria:
  - New collaborator can run project with documented steps.

## Phase 5: Productization and UI (Lower Priority)

### Issue 15: Add simple user interface for pasted screenshots
- Goal: Let users paste screenshots and run segmentation/count quickly.
- Tasks:
  - Maintain `app.py` Streamlit app.
  - Support paste/upload input.
  - Show segmented overlay, binary mask, and vehicle count.
- Acceptance criteria:
  - User can paste image and get result in one click.

### Issue 16: Connect UI to refactored pipeline modules
- Goal: Ensure UI reuses production logic and not notebook code.
- Tasks:
  - Route UI inference through `src/segmentation_project` modules.
  - Keep model configuration externalized in config.
- Acceptance criteria:
  - UI and notebook share the same core code path.

### Issue 17: Improve UI for reliability and reporting
- Goal: Make interface useful for demonstrations and future handoff.
- Tasks:
  - Add better error handling for invalid files/model paths.
  - Add per-class counts and confidence controls.
  - Add export of results (mask, annotated image, JSON metadata).
- Acceptance criteria:
  - UI handles failures gracefully and provides reproducible outputs.
