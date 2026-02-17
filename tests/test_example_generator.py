"""Unit tests for example_generator multi-provider LLM support."""
import sys

sys.path.insert(0, ".")  # noqa: E402

from example_generator import (
    _is_text_completion_model,
    convert_results,
    convert_results_from_json,
)


class TestIsTextCompletionModel:
    """Test model type detection for prompt vs chat completion."""

    def test_text_davinci_003(self):
        assert _is_text_completion_model("text-davinci-003") is True

    def test_text_davinci_002(self):
        assert _is_text_completion_model("text-davinci-002") is True

    def test_text_curie_babbage_ada(self):
        assert _is_text_completion_model("text-curie-001") is True
        assert _is_text_completion_model("text-babbage-001") is True
        assert _is_text_completion_model("text-ada-001") is True

    def test_openai_prefixed_text_completion(self):
        assert _is_text_completion_model("openai/text-davinci-003") is True

    def test_gpt_chat_models_not_text_completion(self):
        assert _is_text_completion_model("gpt-4") is False
        assert _is_text_completion_model("gpt-3.5-turbo") is False
        assert _is_text_completion_model("gpt-4o") is False
        assert _is_text_completion_model("openai/gpt-4") is False

    def test_anthropic_models_not_text_completion(self):
        assert _is_text_completion_model("claude-3-opus") is False
        assert _is_text_completion_model("anthropic/claude-3-5-sonnet") is False

    def test_gemini_models_not_text_completion(self):
        assert _is_text_completion_model("gemini/gemini-pro") is False
        assert _is_text_completion_model("google/gemini-1.5-flash") is False


class TestConvertResults:
    """Tests for convert_results (text 'index: score' format)."""

    def test_standard_format(self):
        """Parses '1: 5\\n2: 4\\n3: 3' into [5, 4, 3]."""
        text = "1: 5\n2: 4\n3: 3"
        assert convert_results(text, "col") == [5, 4, 3]

    def test_single_digit_scores(self):
        """Single-digit scores parse correctly."""
        text = "1: 1\n2: 5\n3: 3"
        assert convert_results(text, "col") == [1, 5, 3]

    def test_two_digit_scores(self):
        """Two-digit scores parse correctly (e.g. scale 1-10)."""
        text = "1: 10\n2: 9\n3: 7"
        assert convert_results(text, "col") == [10, 9, 7]

    def test_extra_whitespace(self):
        """Leading/trailing whitespace and blank lines handled."""
        text = "  1: 5  \n\n  2: 4  \n  3: 3  "
        assert convert_results(text, "col") == [5, 4, 3]

    def test_invalid_line_yields_empty(self):
        """Non-parseable lines yield empty string in list."""
        text = "1: 5\n2: x\n3: 4"
        assert convert_results(text, "col") == [5, "", 4]


class TestConvertResultsFromJson:
    """Tests for convert_results_from_json (structured output format)."""

    def test_valid_single_block(self):
        """Parses {'scores': [5, 4, 3]} into [5, 4, 3]."""
        text = '{"scores": [5, 4, 3]}'
        assert convert_results_from_json(text, "col") == [5, 4, 3]

    def test_valid_multiple_blocks(self):
        """Parses multiple JSON objects (batches) and concatenates scores."""
        text = '{"scores": [5, 4]}\n{"scores": [3, 2]}'
        assert convert_results_from_json(text, "col") == [5, 4, 3, 2]

    def test_invalid_json_returns_none(self):
        """Invalid JSON returns None."""
        assert convert_results_from_json("not json", "col") is None
        assert convert_results_from_json("{invalid}", "col") is None

    def test_missing_scores_key_returns_none(self):
        """JSON without 'scores' key returns None."""
        assert convert_results_from_json('{"data": [1,2,3]}', "col") is None

    def test_empty_scores_returns_none(self):
        """Empty scores array returns None (treated as parse failure)."""
        text = '{"scores": []}'
        result = convert_results_from_json(text, "col")
        assert result is None
