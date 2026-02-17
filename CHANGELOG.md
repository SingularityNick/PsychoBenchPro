# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **uv package manager**: Project now uses [uv](https://docs.astral.sh/uv/) for dependency management.
  - `pyproject.toml` with PEP 621 metadata and dependencies: numpy, scipy, matplotlib, pandas, requests, openai, tenacity, tqdm.
  - `uv.lock` for reproducible installs.
  - `.python-version` set to 3.11 for consistent interpreter selection.
- **README**: New "Setup" section with uv install steps and run instructions (`uv sync`, `uv run python run_psychobench.py ...`).

### Changed

- **.gitignore**: Removed `.python-version` from ignore list so the project’s Python version is tracked in the repo.
