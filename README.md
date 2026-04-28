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



                                                    
                                                    
                                                    
                                                    
                                                 
                               

                                                
