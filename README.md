# Developing Software as a Product  (V5_15)

![CI](https://github.com/ogarmic1landr/developing_software_as_product/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/ogarmic1landr/developing_software_as_product/actions/workflows/release.yml/badge.svg)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![Architecture](https://img.shields.io/badge/architecture-modularized-purple)
![Models](https://img.shields.io/badge/models-SAM%20%7C%20YOLOv8-orange)



## Project Title
Collaborative Refactoring of a Vehicle Segmentation Project from Notebook to Software Product


## Collaborators
**Simon Schmid**,
**Alex Filo**  and
**Michael Ogar**


## Organization
Zurich University of Applied Science (ZHAW)
Department of Applied Computational Life Science


## Module Coordinator
Julija Pecerska


## Date
May 2026


## Project Aim
The goal of this project is to transition an existing vehicle segmentation Jupyter notebook from an academic research project into a maintainable and collaboratively developed software product.
This transformation involves code refactoring, modularization, improved documentation, possible performance optimization, and the introduction of version control and structured collaboration workflows using GitHub.

## State at the start

The original project was developed as a single Jupyter notebook containing all code for data loading, preprocessing, segmentation, and visualization. The notebook was designed for experimentation and research purposes, with hardcoded paths making it difficult to run in another machine, lacked testing and several duplicated logic across different sections.

[Original Notebook](https://github.com/ogarmic1landr/developing_software_as_product/blob/main/notebook.ipynb)

While suitable for experimentation, the monolithic structure limits reusability, scalability, and maintainability. To address this, the project is being refactored into a modular Python based codebase.

The screenshot below illustrates an example of the original implementation, where image loading is performed using hardcoded (image_folder = "images") file paths within the notebook.


![Initial notebook structure](assets/screenshots/Screenshot%20(1192).png)


## Limitations of the Original Notebook

- Existed a single Jupyter notebook without a GitHub repository 
- Monolithic notebook structure with no clear modular separation
- Limited reusability and scalability
- Hardcoded file paths
- Lack of testing and validation for individual components
- Duplication of logic across different sections of the notebook
- Designed for experimentation rather than maintainable software development


## Project Objectives

- Refactor and modularize the existing code

- Consider runtime performance optimizations for the codebase

- Enhance documentation and code readability

- Improve maintainability through clearer structure and naming conventions

- Structure the project for collaborative development using GitHub

- Apply best practices in software product engineering



## Refactoring Plan

To improve the structure of the project, the notebook will be decomposed into modular Python components:

- Image data loading module → Handles image paths and loading
- Image Preprocessing module → Handles image resizing, gray-scaling, and other transformations
- Segmentation module → Runs YOLO/SAM models
- Visualization module → Handles plotting and output display

These modules will be saved as separate .py files, allowing them to be imported into other scripts or components of the project. This enables code reuse across the pipeline and supports a more scalable and maintainable software design.


## Planned Refactor Tracking

Use the issue roadmap in:
[GitHub Issues Roadmap](https://github.com/ogarmic1landr/developing_software_as_product/blob/main/GITHUB_ISSUES_ROADMAP.md)

It provides a chronological GitHub issue sequence to refactor the notebook into a modular,
clean, and maintainable software product.

## Recommended Development Environment

The project has been tested and developed using the following IDEs:

- Visual Studio Code (VS Code)
- PyCharm

but can still run in any Python environment with the required dependencies installed.

### Core Dependencies

- Python
- Jupyter Notebook
- OpenCV
- NumPy
- Matplotlib
- Meta Segment Anything Model (SAM)
- YOLOv8 Segmentation
- Git and GitHub

## Repository Setup

### Clone the Repository

Clone the repository using Git:


```bash
git clone https://github.com/ogarmic1landr/developing_software_as_product.git
```


## Environment Setup

This project supports both Conda-based environments and standard Python virtual environments.

#### Step 1 — Install Conda

Install either:

* Miniconda
* or Anaconda

The Conda environment configuration file can be found here:

* [`environment.yml`](https://github.com/ogarmic1landr/developing_software_as_product/blob/main/environment.yml)

  The Python environment can also be set up using
* [`requirements.txt`](https://github.com/ogarmic1landr/developing_software_as_product/blob/main/requirements.txt)
---

### Option 1: Using Conda (Recommended)

This option is recommended for contributors working with machine learning or scientific Python environments, as it ensures package compatibility and reproducibility across systems.


#### Step 2 — Create the environment

From the project root directory, run:

`conda env create -f environment.yml`

on your terminal. This will create a new Conda environment named `dsp-vehicle-segmentation` with all required dependencies installed.


#### Step 3 — Activate the environment

`conda activate dsp-vehicle-segmentation`

Once activated, all required project dependencies will be available.

---

### Option 2: Using Standard Python (venv)

This option is suitable for contributors who do not use Conda and only have Python installed.

#### Step 1 — Create a virtual environment

Windows:

```bash
python -m venv venv
```

Mac/Linux:

```bash
python3 -m venv venv
```
This creates an isolated Python environment for the project.

#### Step 2 — Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```
Mac/Linux:

```bash
source venv/bin/activate
```
After activation, your terminal should display the virtual environment name.

#### Step 3 — Install required dependencies

```bash
pip install -r requirements.txt
```

This installs all runtime dependencies required for the project.

---

## Debugging and Running the Project

The project can also be executed and debugged directly from the IDE using the configured Python interpreter or Conda environment.


## Collaboration Workflow
All changes must be made in feature branches

Pull Requests are required before merging

Minimum of 1 approvals from collaborators required


## Refactored Repository Structure

The original monolithic notebook has been reorganized into a modular and maintainable software project structure.

The repository is organized into reusable components for:
- data loading,
- preprocessing,
- segmentation,
- visualization,
- and model evaluation.

The current project structure is shown below:


```text
developing_software_as_product/
│
├── assets/
│   └── screenshots/
│
├── docs/
|
├── images/
│
├── masks_auto/
│
├── masks_manual/      
│
├── src/
│   └── segmentation_project/
│       │
│       ├── data/
│       │   ├── loader.py
│       │   └── testing_loader.ipynb
│       │
│       ├── preprocessing/
│       │   ├── preprocessing.py
│       │   └── test_preprocessing.ipynb
│       │
│       ├── ground_truth/
│       │   ├── overlay.py
│       │   └── test_overlay.ipynb
│       │
│       ├── segmentation_models/
│       │   ├── sam_segmenter.py
│       │   ├── yolo_segmenter.py
│       │   ├── test_sam.ipynb
│       │   └── test_yolo.ipynb
│       │
│       └── model_evaluation/
│           ├── evaluation.py
│           └── test_evaluation.ipynb
│
├── .github/
|── tests/
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── environment.yml
├── GITHUB_ISSUES_ROADMAP.md
├── LICENSE
├── notebook.ipynb
├── pyproject.toml
├── README.md
└── requirements.txt
```


## Running the segmentation experiments pipeline.

The notebooks below represent the recommended execution order for reproducing the segmentation experiments.

---

### 1. Dataset Loading

Notebook:

`src/segmentation_project/data/testing_loader.ipynb`

Purpose:
- validate image loading,
- verify dataset paths,
- load manual ground truth masks.

---

### 2. Image Preprocessing

Notebook:

`src/segmentation_project/preprocessing/test_preprocessing.ipynb`

Purpose:
- resize images,
- convert images to grayscale,
- apply CLAHE enhancement,
- validate preprocessing operations.

---

### 3. Ground Truth Visualization

Notebook:

`src/segmentation_project/ground_truth/test_overlay.ipynb`

Purpose:
- overlay segmentation masks onto images,
- verify alignment between masks and images,
- visualize preprocessing outputs.

---

### 4. YOLOv8 Segmentation

Notebook:

`src/segmentation_project/segmentation_models/test.yolo.ipynb`

Purpose:
- run YOLOv8 segmentation experiments,
- visualize segmentation predictions,
- test automatic vehicle segmentation.

---

### 5. SAM Segmentation

Notebook:

`src/segmentation_project/segmentation_models/test_sam.ipynb`

Purpose:
- generate automatic segmentation masks using the Segment Anything Model (SAM),
- save predicted masks into:

`masks_auto/`

Important Notes:
- SAM inference is computationally expensive on CPU-only systems.
- Processing a single image may take several minutes depending on:
  - image resolution,
  - available RAM,
  - and hardware acceleration.
- Generated masks already included in `masks_auto/` do not need to be regenerated unless:
  - new images are added,
  - preprocessing settings are modified,
  - or segmentation outputs need to be regenerated.

---

### 6. Model Evaluation

Notebook:

`src/segmentation_project/model_evaluation/test_evaluation.ipynb`

Purpose:
- compare predicted masks against manual ground truth masks,
- compute:
  - Dice Similarity Coefficient (DSC),
  - Intersection over Union (IoU),
- generate evaluation plots and summary statistics.



## Ground Truth Masks

The manual segmentation masks stored in:

`masks_manual/`

were created manually using the GNU Image Manipulation Program (GIMP).

These manually annotated masks serve as the ground truth reference for evaluating the automatic segmentation outputs generated by  SAM.

The ground truth masks are used during evaluation to compute:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)

This allows quantitative comparison between the predicted segmentation masks and the manually annotated vehicle regions.


## Development Tooling

The repository includes additional software engineering tooling to support maintainability and collaborative development, including:

- GitHub Actions CI/CD workflows
- Ruff linting and formatting
- Pre-commit hooks
- Git-based collaborative workflows


## Contribution Guidelines

Please review the repository contribution rules before opening Pull Requests:

`CONTRIBUTING.md`


## Future Work and Improvements

Several future improvements could further extend the project beyond the current notebook refactoring and modularizing into a software product supported by collaborative GitHub workflows

### Streamlit User Interface

A lightweight Streamlit interface was explored during development as a potential frontend layer for interactive segmentation testing and visualization.

Future work could expand this interface into a more complete application supporting:
- interactive image upload,
- live segmentation visualization,
- evaluation metric dashboards,
- and deployment-ready workflows.



## License
This project is licensed under the MIT License. See the LICENSE file for details.
