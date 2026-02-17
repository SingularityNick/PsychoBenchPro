# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Hydra configuration**: Integrated [Hydra](https://hydra.cc/) for configuration and CLI overrides.
  - Default config in `conf/config.yaml` (model, questionnaire, shuffle_count, test_count, mode, significance_level, openai_key, etc.).
  - Run from project root with overrides, e.g. `python run_psychobench.py model=gpt-4 questionnaire=BFI,EPQ-R`.
  - `example_generator` and `run_psychobench` use a unified run config object.
  - `.gitignore` updated to exclude Hydra output directories (e.g. `.hydra/`, `multirun/`).
- **CI pipeline**: GitHub Actions workflow (`.github/workflows/tests.yml`) runs pytest on push to `main` and on pull requests, using uv and Python 3.11.
- **uv package manager**: Project now uses [uv](https://docs.astral.sh/uv/) for dependency management.
  - `pyproject.toml` with PEP 621 metadata and dependencies: numpy, scipy, matplotlib, pandas, requests, openai, tenacity, tqdm, loguru.
  - `uv.lock` for reproducible installs.
  - `.python-version` set to 3.11 for consistent interpreter selection.
- **README**: New "Setup" section with uv install steps and run instructions (`uv sync`, `uv run python run_psychobench.py ...`).
- **Pytest unit tests**: Test suite for high-priority logic in `utils.py`, `example_generator.py`, and Hydra config.
  - `tests/conftest.py`: Shared fixtures (minimal questionnaires, CSV fixtures for `convert_data`), project root on `sys.path` for imports.
  - `tests/test_utils.py`: Tests for `get_questionnaire`, `convert_data`, `compute_statistics`, `hypothesis_testing`, `parsing`.
  - `tests/test_example_generator.py`: Tests for `convert_results`.
  - `tests/test_hydra_config.py`: Tests for Hydra config composition (Compose API, default keys, overrides, comma-separated questionnaire) and for `run_psychobench` behavior when called with composed config (e.g. questionnaire validation).
  - Run with `pytest tests -v` from project root (or `uv run pytest tests -v`).
- **README**: "Tests" section with install and run instructions (`pytest tests -v`, optional coverage).
- **Loguru logging**: Integrated [loguru](https://github.com/Delgan/loguru) for improved error handling and logging across `example_generator.py`, `run_psychobench.py`, and `utils.py`.

### Changed

- **Hydra output directory**: Each run now creates a timestamped output directory under `results/` (e.g. `results/2026-02-17/12-16-36/`) following Hydra’s default pattern. Hydra config (`.hydra/`), logs, and PsychoBench outputs (CSV, MD, figures) are written into that directory. When running under the Compose API (e.g. unit tests), output falls back to `results/` so existing tests pass unchanged.
- **Dependency management**: Removed `requirements.txt` and `requirements-dev.txt`; dependencies are managed solely via `pyproject.toml` and uv. Python version is set in `pyproject.toml` (e.g. `>=3.11`).
- **.gitignore**: Removed `.python-version` from ignore list so the project's Python version is tracked in the repo.
