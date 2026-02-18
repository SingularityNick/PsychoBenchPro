# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- **Structured output schema**: Response schema now uses required `question_index` and `score` per answer (Pydantic `AnswerItem`) so models (e.g. Gemini) no longer return empty objects. Parser still returns the same `list[int]` in question order and accepts the old format `[{"1": 5}, {"2": 3}]` for backward compatibility.

### Added

- **Optional structured output**: New config option `use_structured_output` (default: `false`) enables JSON-mode structured output for LLM responses. When enabled, the LLM is asked to return answers as a JSON object conforming to a Pydantic schema (`QuestionnaireResponse`), which is far more reliable than the legacy text-based "last digit per line" parser. Requires a model that supports structured output / JSON mode (most modern models do). Falls back gracefully to the legacy parser if JSON parsing fails. Enable with `use_structured_output=true` on the CLI or in `conf/config.yaml`.
- **Benchmark config**: `conf/benchmark.yaml` for running a Hydra multirun over multiple models without listing them on the CLI. Run with `uv run python run_psychobench.py --config-name benchmark`; edit `hydra.sweeper.params.model` in that file to change the model list. README updated with a "Benchmark config" subsection under Configuration.
- **Multi-provider LLM support**: Model must use a provider prefix (e.g. `openai/gpt-4`, `anthropic/claude-3-5-sonnet`, `gemini/gemini-2.0-flash`, `ollama/llama2`). Config option `allowed_providers` (default: `gemini`, `anthropic`, `openai`, `ollama`) restricts which providers can be used. Validation runs at startup so invalid or unsupported models fail fast.
- **Ollama provider**: Support for local models via [Ollama](https://ollama.ai/) using the `ollama/` prefix (e.g. `ollama/llama2`, `ollama/deepseek-r1:latest`). No API key required when using Ollama.
- **Model name for file naming**: Response and prompt filenames use the model name without the provider prefix (e.g. `gpt-4-BFI-shuffle0.txt`) for cleaner output.
- **Optional custom API base**: Config option `api_base` for custom endpoints (e.g. Azure, OpenAI-compatible proxies); empty by default.
- **`.env.example`**: Example file with placeholders for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`; copy to `.env` and fill in keys (LiteLLM reads from environment).
- **Hydra configuration**: Integrated [Hydra](https://hydra.cc/) for configuration and CLI overrides.
  - Default config in `conf/config.yaml` (model, questionnaire, shuffle_count, test_count, mode, significance_level, allowed_providers, api_base, etc.).
  - Run from project root with overrides, e.g. `python run_psychobench.py model=openai/gpt-4 questionnaire=BFI,EPQ-R`.
  - `example_generator` and `run_psychobench` use a unified run config object.
  - `.gitignore` updated to exclude Hydra output directories (e.g. `.hydra/`, `multirun/`).
- **CI pipeline**: GitHub Actions workflow (`.github/workflows/tests.yml`) runs pytest on push to `main` and on pull requests, using uv and Python 3.11.
- **uv package manager**: Project now uses [uv](https://docs.astral.sh/uv/) for dependency management.
  - `pyproject.toml` with PEP 621 metadata and dependencies: numpy, scipy, matplotlib, pandas, requests, litellm, tenacity, tqdm, loguru.
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

- **LLM API integration**: Refactored from OpenAI SDK to [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM API access. LiteLLM supports OpenAI and many other providers (Anthropic, Cohere, etc.), making it easier to use different LLM providers.
- **API key configuration**: API keys are read from environment variables by LiteLLM (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). Removed `openai_key` from config in favor of env-based keys for all providers.
- **Model validation**: Added provider-prefix validation and LiteLLM model-registry check before running so unknown or disallowed models are rejected early.
- **Hydra output directory**: Each run now creates a timestamped output directory under `results/` (e.g. `results/2026-02-17/12-16-36/`) following Hydra’s default pattern. Hydra config (`.hydra/`), logs, and PsychoBench outputs (CSV, MD, figures) are written into that directory. When running under the Compose API (e.g. unit tests), output falls back to `results/` so existing tests pass unchanged.
- **Dependency management**: Removed `requirements.txt` and `requirements-dev.txt`; dependencies are managed solely via `pyproject.toml` and uv. Python version is set in `pyproject.toml` (e.g. `>=3.11`).
- **.gitignore**: Removed `.python-version` from ignore list so the project's Python version is tracked in the repo.
- **CI (Tests workflow)**: Tests no longer run when only documentation files change. Added `paths-ignore` for `pull_request` and `push` to skip runs when changes are limited to `README*`, `CHANGELOG*`, `docs/**`, and `**.md`.
