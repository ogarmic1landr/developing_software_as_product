# Contributing Guidelines

This repository follows a collaborative software engineering workflow designed to support maintainability, reproducibility, modularity, and structured development practices.

The project originally began as a research-oriented Jupyter notebook and has since been refactored into a modular software product architecture. Contributors are encouraged to follow the established workflow and coding standards described below.

# Development Workflow

## Branching Rules

Direct commits to `main` are not allowed.

All development should be performed through feature branches followed by Pull Request (PR).


## Pull Requests

Before merging:
- Open a Pull Request (PR)
- Provide a clear PR title
- Describe the changes implemented in the PR description
- Ensure all CI checks pass


## Merging
- At least one approval from a project collaborator is required before merging
- The PR author or a project collaborator can merge the PR once approved and all checks have passed

# Coding Standards
- Follow the existing code style and conventions used in the project
- Write clear and concise code with appropriate comments or docstrings where necessary
- Ensure codes is modular and reusable where possible
- Include tests for new features or bug fixes when applicable
- Use `.gitignore` to manage untracked files appropriately.


`main` branch should remain stable and runnable at all times.