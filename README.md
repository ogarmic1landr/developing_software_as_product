# Developing Software as a Product  (V5_15)


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
This transformation involves code refactoring, modularization, improved documentation, performance optimization, and the introduction of version control and structured collaboration workflows using GitHub.


## Project Objectives
- Refactor and modularize the existing code

- Improve runtime performance and scalability

- Enhance documentation and code readability

- Improve maintainability through clearer structure and naming conventions

- Structure the project for collaborative development using GitHub

- Apply best practices in software product engineering


## Optional UI Demo (Paste Screenshot -> Segment + Count)
This repository now includes a lightweight Streamlit UI for quick testing on screenshots
(including Google Maps screenshots pasted into the uploader).

### What it does
- Loads an uploaded or pasted image
- Runs YOLOv8 segmentation/detection
- Highlights detected vehicle regions
- Counts vehicles (car, bus, truck classes)

### Run
1. Install required runtime packages (if missing in your environment):
	- `pip install -r requirements-ui.txt`
2. Start the app from project root:
	- `streamlit run app.py`
3. In the browser UI, paste or upload a screenshot and click **Run segmentation**.


## Planned Refactor Tracking
Use the issue roadmap in:
- `GITHUB_ISSUES_ROADMAP.md`

It provides a chronological GitHub issue sequence to refactor the notebook into a modular,
clean, and maintainable software product.


## Technologies and Tools
- Python

- Jupyter Notebook

- OpenCV

- Numpy

- Matplotlib

- Meta segment anything

- YOLOv8m segmentation model

- Git and GitHub


## Environment Setup

This project supports both Conda-based environments and standard Python virtual environments.

The Conda environment configuration file can be found here:

* `environment.yml`
  https://github.com/ogarmic1landr/developing_software_as_product/blob/main/environment.yml

---

### Option 1: Using Conda (Recommended)

This option is recommended for contributors working with machine learning or scientific Python environments, as it ensures package compatibility and reproducibility across systems.

#### Step 1 — Install Conda

Install either:

* Miniconda
* or Anaconda

#### Step 2 — Create the environment

From the project root directory, run:

*conda env create -f environment.yml*

on your terminal. This will create a new Conda environment named `dsp-vehicle-segmentation` with all required dependencies installed.


#### Step 3 — Activate the environment

*conda activate dsp-vehicle-segmentation*

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

## Running the Application

After the environment has been activated and dependencies installed, run the Streamlit application from the project root:

```bash
streamlit run app.py
```

The application will open in your default browser and allow image upload and vehicle segmentation testing.


## Collaboration Workflow
All changes must be made in feature branches

Pull Requests are required before merging

Minimum of 1 approvals from collaborators required


## Current codebase
This project was initially a Jupyter notebook for vehicle image segmentation. The notebook contains all core logic within a single file, including data loading, Image preprocessing steps, Testing several Image segmentation models, and visualization.

While suitable for experimentation, the monolithic structure limits reusability, scalability, and maintainability. To address this, the project is being refactored into a modular Python based codebase.

The screenshot below illustrates an example of the original implementation, where image loading is performed using hardcoded (image_folder = "images") file paths within the notebook.


![Initial notebook structure](assets/screenshots/Screenshot%20(1192).png)


## Refactoring Plan

To improve the structure of the project, the notebook will be decomposed into modular Python components:

- Image data loading module → Handles image paths and loading
- Image Preprocessing module → Handles image resizing, gray-scaling, and other transformations
- Segmentation module → Runs YOLO/SAM models
- Visualization module → Handles plotting and output display

These modules will be saved as separate .py files, allowing them to be imported into other scripts or components of the project. This enables code reuse across the pipeline and supports a more scalable and maintainable software design.


## Current Limitations of the Original Notebook

The original vehicle segmentation project was developed as a research-oriented Jupyter notebook.

While functional, it has several limitations when viewed as a software product this include:

- Monolithic notebook structure with no clear modular separation
- Limited reusability and scalability
- Hard coded parameters and file paths
- Minimal documentation
- Designed for experimentation rather than maintainable software development

## License
This project is licensed under the MIT License. See the LICENSE file for details.
