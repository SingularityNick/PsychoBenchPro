# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **uv package manager**: Project now uses [uv](https://docs.astral.sh/uv/) for dependency management.
  - `pyproject.toml` with PEP 621 metadata and dependencies: numpy, scipy, matplotlib, pandas, requests, openai, tenacity, tqdm, loguru.
  - `uv.lock` for reproducible installs.
  - `.python-version` set to 3.11 for consistent interpreter selection.
- **README**: New "Setup" section with uv install steps and run instructions (`uv sync`, `uv run python run_psychobench.py ...`).
- **Pytest unit tests**: Test suite for high-priority logic in `utils.py` and `example_generator.py`.
  - `tests/conftest.py`: Shared fixtures (minimal questionnaires, CSV fixtures for `convert_data`), project root on `sys.path` for imports.
  - `tests/test_utils.py`: Tests for `get_questionnaire`, `convert_data`, `compute_statistics`, `hypothesis_testing`, `parsing`.
  - `tests/test_example_generator.py`: Tests for `convert_results`.
  - Run with `pytest tests -v` from project root (or `uv run pytest tests -v`).
- **README**: "Tests" section with install and run instructions (`pytest tests -v`, optional coverage).
- **Loguru logging**: Integrated [loguru](https://github.com/Delgan/loguru) for improved error handling and logging across `example_generator.py`, `run_psychobench.py`, and `utils.py`.

### Changed

- **Dependency management**: Removed `requirements.txt` and `requirements-dev.txt`; dependencies are managed solely via `pyproject.toml` and uv. Python version is set in `pyproject.toml` (e.g. `>=3.11`).
- **.gitignore**: Removed `.python-version` from ignore list so the project's Python version is tracked in the repo.
