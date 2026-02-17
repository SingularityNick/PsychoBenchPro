"""Unit tests for example_generator.py — convert_results and multi-provider support."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm

from example_generator import _configure_litellm, chat, convert_results

# ---------------------------------------------------------------------------
# convert_results
# ---------------------------------------------------------------------------

class TestConvertResults:
    """Tests for convert_results."""

    def test_normal_input(self):
        result = convert_results("1. 5\n2. 3\n3. 4", "col")
        assert result == [5, 3, 4]

    def test_strips_whitespace(self):
        result = convert_results("  1. 5  \n  2. 3  \n  3. 4  ", "col")
        assert result == [5, 3, 4]

    def test_single_line(self):
        result = convert_results("1. 7", "col")
        assert result == [7]

    def test_empty_lines_skipped(self):
        result = convert_results("1. 5\n\n2. 3\n", "col")
        assert result == [5, 3]

    def test_invalid_input_returns_empty_strings(self):
        result = convert_results("not a number", "col")
        assert all(elem == "" for elem in result)


# ---------------------------------------------------------------------------
# _configure_litellm
# ---------------------------------------------------------------------------

class TestConfigureLitellm:
    """Tests for _configure_litellm API key and base URL resolution."""

    def setup_method(self):
        """Reset litellm globals before each test."""
        litellm.api_key = None
        litellm.api_base = None

    def test_api_key_takes_priority(self, monkeypatch):
        """api_key should be used over openai_key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        run = SimpleNamespace(api_key="sk-generic", openai_key="sk-openai", api_base=None)
        _configure_litellm(run)
        assert litellm.api_key == "sk-generic"

    def test_openai_key_fallback(self, monkeypatch):
        """When api_key is empty, openai_key should be used as fallback."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        run = SimpleNamespace(api_key="", openai_key="sk-openai-fallback", api_base=None)
        _configure_litellm(run)
        assert litellm.api_key == "sk-openai-fallback"

    def test_no_keys_does_not_set(self):
        """When both keys are empty, litellm.api_key stays None."""
        run = SimpleNamespace(api_key="", openai_key="", api_base=None)
        _configure_litellm(run)
        assert litellm.api_key is None

    def test_api_base_set(self):
        """api_base should be forwarded to litellm."""
        run = SimpleNamespace(api_key="", openai_key="", api_base="http://localhost:8000")
        _configure_litellm(run)
        assert litellm.api_base == "http://localhost:8000"

    def test_api_base_none(self):
        """When api_base is null/None, litellm.api_base should remain None."""
        run = SimpleNamespace(api_key="", openai_key="", api_base=None)
        _configure_litellm(run)
        assert litellm.api_base is None

    def test_missing_attrs_handled_gracefully(self):
        """If run config has no api_key/openai_key/api_base attrs, no error is raised."""
        run = SimpleNamespace()
        _configure_litellm(run)
        assert litellm.api_key is None
        assert litellm.api_base is None

    def test_openai_env_var_set_as_default(self, monkeypatch):
        """effective key is set as OPENAI_API_KEY env default via setdefault."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        run = SimpleNamespace(api_key="sk-test-env", openai_key="", api_base=None)
        _configure_litellm(run)
        assert os.environ.get("OPENAI_API_KEY") == "sk-test-env"

    def test_existing_openai_env_var_not_overwritten(self, monkeypatch):
        """setdefault should not overwrite an existing OPENAI_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-existing")
        run = SimpleNamespace(api_key="sk-new", openai_key="", api_base=None)
        _configure_litellm(run)
        assert os.environ["OPENAI_API_KEY"] == "sk-existing"


# ---------------------------------------------------------------------------
# chat() — model-agnostic routing
# ---------------------------------------------------------------------------

class TestChatMultiProvider:
    """Verify that chat() passes model identifiers directly to litellm.completion."""

    def _mock_response(self, content="1: 5"):
        """Build a minimal mock response matching litellm's structure."""
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @patch("example_generator.litellm")
    def test_openai_model(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_response("ok")
        result = chat("gpt-4", [{"role": "user", "content": "hi"}], delay=0)
        assert result == "ok"
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4"

    @patch("example_generator.litellm")
    def test_anthropic_model(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_response("claude says hi")
        result = chat("claude-3-5-sonnet-20241022", [{"role": "user", "content": "hi"}], delay=0)
        assert result == "claude says hi"
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["model"] == "claude-3-5-sonnet-20241022"

    @patch("example_generator.litellm")
    def test_gemini_model(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_response("gemini says hi")
        result = chat("gemini/gemini-1.5-pro", [{"role": "user", "content": "hi"}], delay=0)
        assert result == "gemini says hi"
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["model"] == "gemini/gemini-1.5-pro"

    @patch("example_generator.litellm")
    def test_custom_model_with_api_base(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_response("custom")
        result = chat(
            "openai/my-model",
            [{"role": "user", "content": "hi"}],
            delay=0,
            api_base="http://localhost:8000",
        )
        assert result == "custom"
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/my-model"
        assert call_kwargs.kwargs["api_base"] == "http://localhost:8000"

    @patch("example_generator.litellm")
    def test_api_base_omitted_when_none(self, mock_litellm):
        mock_litellm.completion.return_value = self._mock_response("no base")
        chat("gpt-4", [{"role": "user", "content": "hi"}], delay=0, api_base=None)
        call_kwargs = mock_litellm.completion.call_args
        assert "api_base" not in call_kwargs.kwargs

    @patch("example_generator.litellm")
    def test_multiple_choices(self, mock_litellm):
        c1, c2 = MagicMock(), MagicMock()
        c1.message.content = "a"
        c2.message.content = "b"
        resp = MagicMock()
        resp.choices = [c1, c2]
        mock_litellm.completion.return_value = resp
        result = chat("gpt-4", [{"role": "user", "content": "hi"}], n=2, delay=0)
        assert result == ["a", "b"]
