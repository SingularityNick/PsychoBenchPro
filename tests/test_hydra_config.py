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
        assert hasattr(cfg, "name_exp")
        assert hasattr(cfg, "significance_level")
        assert hasattr(cfg, "mode")
        assert hasattr(cfg, "openai_key")
        assert hasattr(cfg, "api_key")
        assert hasattr(cfg, "api_base")
        assert cfg.mode == "auto"
        assert cfg.significance_level == 0.01
        assert OmegaConf.is_config(cfg)

    def test_compose_overrides_apply(self):
        """CLI-style overrides are applied to composed config."""
        overrides = [
            "model=my-model",
            "questionnaire=BFI",
            "mode=generation",
            "shuffle_count=0",
        ]
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        assert cfg.model == "my-model"
        assert cfg.questionnaire == "BFI"
        assert cfg.mode == "generation"
        assert cfg.shuffle_count == 0

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

    def test_run_psychobench_builds_run_config_and_calls_generator(self):
        """Composed cfg flows to run_psychobench; mock generator receives correct run paths."""
        calls = []

        def mock_gen(questionnaire, run):
            calls.append((questionnaire, run))

        overrides = [
            "questionnaire=BFI",
            "mode=testing",
            "model=test-model",
        ]
        with initialize(version_base=None, config_path=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        run_psychobench(cfg, mock_gen)
        assert len(calls) >= 1
        questionnaire, run = calls[0]
        assert questionnaire["name"] == "BFI"
        assert run.model == "test-model"
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
            "model=test-gen",
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
