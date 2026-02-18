"""Unit tests for Hydra config composition and run_psychobench with composed config.

Uses Hydra's Compose API (initialize + compose) as recommended in
https://hydra.cc/docs/advanced/unit_testing/ — no @hydra.main() in tests.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from utils import run_psychobench

# Config path relative to this test file (tests/test_hydra_config.py) -> ../conf
CONFIG_PATH = "../conf"
CONFIG_NAME = "config"
# Project root and path to config file (single source of truth for allowed_providers etc.)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(_ROOT, "conf", "config.yaml")


class TestComposeAPI:
    """Config composition via Hydra Compose API."""

    def test_compose_loads_default_config(self):
        """Compose API loads conf/config.yaml and default keys exist."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME)
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "questionnaire")
        assert hasattr(cfg, "shuffle_count")
        assert hasattr(cfg, "test_count")
        assert hasattr(cfg, "significance_level")
        assert hasattr(cfg, "mode")
        assert hasattr(cfg, "allowed_providers")
        assert hasattr(cfg, "use_structured_output")
        assert hasattr(cfg, "structured_output_notes")
        assert hasattr(cfg, "batch_size")
        assert hasattr(cfg, "max_parse_failure_retries")
        assert cfg.mode == "auto"
        assert cfg.batch_size == 30
        assert cfg.max_parse_failure_retries == 3
        assert cfg.significance_level == 0.01
        # use_structured_output, structured_output_notes, and allowed_providers
        # must match conf/config.yaml (no hardcoded values)
        expected = OmegaConf.load(CONFIG_FILE)
        assert cfg.use_structured_output == expected.use_structured_output
        assert cfg.structured_output_notes == expected.structured_output_notes
        assert list(cfg.allowed_providers) == list(expected.allowed_providers)
        assert OmegaConf.is_config(cfg)

    def test_structured_output_notes_defaults_true(self):
        """structured_output_notes defaults to true in config."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME)
        assert cfg.structured_output_notes is True

    def test_structured_output_notes_override_false(self):
        """structured_output_notes can be overridden to false."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["structured_output_notes=false"],
            )
        assert cfg.structured_output_notes is False

    def test_compose_overrides_apply(self):
        """CLI-style overrides are applied to composed config."""
        overrides = [
            "model=openai/my-model",
            "questionnaire=BFI",
            "mode=generation",
            "shuffle_count=0",
        ]
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        assert cfg.model == "openai/my-model"
        assert cfg.questionnaire == "BFI"
        assert cfg.mode == "generation"
        assert cfg.shuffle_count == 0

    def test_compose_batch_size_null_no_batching(self):
        """batch_size=null means no batching (full questionnaire as single batch)."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["batch_size=null"],
            )
        assert cfg.batch_size is None

    def test_compose_batch_size_override(self):
        """batch_size can be overridden to a custom value."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["batch_size=10"],
            )
        assert cfg.batch_size == 10

    def test_compose_max_parse_failure_retries_override(self):
        """max_parse_failure_retries can be overridden to a custom value."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["max_parse_failure_retries=5"],
            )
        assert cfg.max_parse_failure_retries == 5

    def test_compose_max_parse_failure_retries_zero_disables(self):
        """max_parse_failure_retries=0 disables retries (fail on first error)."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["max_parse_failure_retries=0"],
            )
        assert cfg.max_parse_failure_retries == 0

    def test_compose_questionnaire_comma_separated(self):
        """questionnaire=BFI,EPQ-R is preserved when quoted (Hydra treats unquoted comma as sweep)."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["questionnaire='BFI,EPQ-R'"],
            )
        assert cfg.questionnaire == "BFI,EPQ-R"
        parts = [p.strip() for p in cfg.questionnaire.split(",")]
        assert parts == ["BFI", "EPQ-R"]


class TestRunPsychobenchWithComposedConfig:
    """run_psychobench called with config from Compose API."""

    def test_run_psychobench_raises_when_questionnaire_null(self):
        """ValueError when questionnaire is not set."""
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=["questionnaire=null"],
            )
        def mock_generator(q, r):
            return None

        with pytest.raises(ValueError, match="questionnaire must be set"):
            run_psychobench(cfg, mock_generator)

    def test_run_psychobench_raises_when_model_lacks_provider_prefix(self):
        """ValueError when model does not start with an allowed provider prefix."""
        from example_generator import example_generator

        overrides = [
            "questionnaire=BFI",
            "mode=testing",
            "model=gpt-4",
        ]
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        with pytest.raises(ValueError, match="model must start with a provider prefix"):
            run_psychobench(cfg, example_generator)

    def test_run_psychobench_builds_run_config_and_calls_generator(self):
        """Composed cfg flows to run_psychobench; mock generator receives correct run paths."""
        calls = []

        def mock_gen(questionnaire, run):
            calls.append((questionnaire, run))

        overrides = [
            "questionnaire=BFI",
            "mode=testing",
            "model=openai/test-model",
        ]
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        run_psychobench(cfg, mock_gen)
        assert len(calls) >= 1
        questionnaire, run = calls[0]
        assert questionnaire["name"] == "BFI"
        assert run.model == "openai/test-model"
        assert run.testing_file == "results/test-model-BFI.csv"
        assert run.results_file == "results/test-model-BFI.md"
        assert run.figures_file == "test-model-BFI.png"

    def test_run_psychobench_mode_generation_only_creates_testfile(
        self, tmp_path, monkeypatch
    ):
        """Generation-only mode produces expected CSV under tmp_path."""
        minimal_questionnaire = {
            "name": "BFI",
            "prompt": "Test prompt",
            "questions": {"1": "Q1", "2": "Q2"},
            "scale": 5,
            "compute_mode": "AVG",
            "reverse": [],
            "categories": [],
        }

        def fake_get_questionnaire(name):
            assert name == "BFI"
            return minimal_questionnaire

        import utils as utils_mod
        monkeypatch.setattr(utils_mod, "get_questionnaire", fake_get_questionnaire)
        monkeypatch.chdir(tmp_path)

        overrides = [
            "questionnaire=BFI",
            "mode=generation",
            "model=openai/test-gen",
            "shuffle_count=0",
            "test_count=1",
        ]
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

        def mock_generator(q, r):
            return None

        run_psychobench(cfg, mock_generator)

        csv_path = tmp_path / "results" / "test-gen-BFI.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "Prompt:" in content
        assert "order-0" in content
        assert "shuffle0-test0" in content
