"""Unit tests for example_generator multi-provider LLM support and structured output."""
import csv
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

sys.path.insert(0, ".")  # noqa: E402

from example_generator import (
    _build_response_model,
    _build_structured_prompt,
    _is_text_completion_model,
    convert_results,
    convert_results_structured,
    example_generator,
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


# ---------------------------------------------------------------------------
# _build_response_model (dynamic Pydantic model factory)
# ---------------------------------------------------------------------------


class TestBuildResponseModel:
    """Tests for the dynamic Pydantic structured output model."""

    def test_valid_json_parses(self):
        raw = '{"q1": 5, "q2": 3, "q3": 4}'
        model_cls = _build_response_model(3)
        resp = model_cls.model_validate_json(raw)
        assert resp.q1 == 5
        assert resp.q2 == 3
        assert resp.q3 == 4

    def test_missing_field_raises(self):
        raw = '{"q1": 5}'
        model_cls = _build_response_model(2)
        with pytest.raises(ValueError):
            model_cls.model_validate_json(raw)

    def test_non_integer_score_raises(self):
        raw = '{"q1": "high"}'
        model_cls = _build_response_model(1)
        with pytest.raises(ValueError):
            model_cls.model_validate_json(raw)

    def test_score_out_of_range_raises(self):
        raw = '{"q1": 0}'
        model_cls = _build_response_model(1)
        with pytest.raises(ValueError):
            model_cls.model_validate_json(raw)
        raw = '{"q1": 6}'
        with pytest.raises(ValueError):
            model_cls.model_validate_json(raw)

    def test_schema_has_all_required_fields(self):
        """The model generates a JSON schema with all q1..qN fields required."""
        model_cls = _build_response_model(3)
        schema = model_cls.model_json_schema()
        props = schema.get("properties", {})
        assert "q1" in props
        assert "q2" in props
        assert "q3" in props
        assert set(schema.get("required", [])) == {"q1", "q2", "q3"}

    def test_caching_returns_same_class(self):
        assert _build_response_model(5) is _build_response_model(5)


# ---------------------------------------------------------------------------
# convert_results_structured
# ---------------------------------------------------------------------------


class TestConvertResultsStructured:
    """Tests for the structured JSON response parser."""

    def test_valid_format_returns_scores(self):
        raw = '{"q1": 5, "q2": 3, "q3": 4}'
        result = convert_results_structured(raw, 3)
        assert result == [5, 3, 4]

    def test_single_question(self):
        raw = '{"q1": 2}'
        result = convert_results_structured(raw, 1)
        assert result == [2]

    def test_invalid_json_raises(self):
        """Invalid input raises ValidationError."""
        with pytest.raises(ValidationError):
            convert_results_structured("1: 5\n2: 3", 2)

    def test_missing_field_raises(self):
        """Fewer fields than n_questions raises ValidationError."""
        with pytest.raises(ValidationError):
            convert_results_structured('{"q1": 5}', 3)

    def test_scores_are_integers(self):
        raw = '{"q1": 1, "q2": 4, "q3": 3}'
        result = convert_results_structured(raw, 3)
        assert all(isinstance(s, int) for s in result)

    def test_preserves_field_order(self):
        raw = '{"q3": 1, "q1": 5, "q2": 3}'
        result = convert_results_structured(raw, 3)
        assert result == [5, 3, 1]


# ---------------------------------------------------------------------------
# _build_structured_prompt
# ---------------------------------------------------------------------------


class TestBuildStructuredPrompt:
    """Tests for the structured output prompt builder."""

    def test_includes_questions(self):
        q = {"prompt": "Rate these:", "inner_setting": "You are a helper."}
        questions = "1. I am talkative.\n2. I am reserved."
        prompt = _build_structured_prompt(q, questions, 2)
        assert "1. I am talkative." in prompt
        assert "2. I am reserved." in prompt

    def test_includes_json_format_instructions(self):
        q = {"prompt": "Rate these:", "inner_setting": "You are a helper."}
        prompt = _build_structured_prompt(q, "1. Q1", 1)
        assert "JSON" in prompt
        assert "q1" in prompt
        assert "1-5" in prompt

    def test_includes_correct_range(self):
        q = {"prompt": "Rate these:", "inner_setting": "You are a helper."}
        prompt = _build_structured_prompt(q, "1. Q1\n2. Q2\n3. Q3", 3)
        assert "q1" in prompt
        assert "q3" in prompt

    def test_includes_base_prompt(self):
        q = {"prompt": "Please evaluate yourself.", "inner_setting": "You are a helper."}
        prompt = _build_structured_prompt(q, "1. Q1", 1)
        assert "Please evaluate yourself." in prompt


# ---------------------------------------------------------------------------
# convert_results (legacy parser — ensure no regression)
# ---------------------------------------------------------------------------


class TestConvertResultsLegacy:
    """Regression tests for the legacy text parser."""

    def test_standard_format(self):
        text = "1: 5\n2: 3\n3: 7"
        assert convert_results(text, "col") == [5, 3, 7]

    def test_empty_lines_ignored(self):
        text = "1: 5\n\n2: 3\n\n3: 7"
        assert convert_results(text, "col") == [5, 3, 7]

    def test_whitespace_stripped(self):
        text = "  1: 5  \n  2: 3  "
        assert convert_results(text, "col") == [5, 3]

    def test_unparseable_returns_empty_strings(self):
        text = "no scores here\nanother line"
        result = convert_results(text, "col")
        assert all(v == "" for v in result)


# ---------------------------------------------------------------------------
# max_parse_failure_retries integration tests
# ---------------------------------------------------------------------------


def _make_testing_csv(path, num_questions=3):
    """Create a minimal testing CSV for example_generator with the given number of questions."""
    prompt_col = ["Prompt: Rate yourself"] + [f"{i+1}. Q{i+1}" for i in range(num_questions)]
    order_col = ["order-0"] + [str(i + 1) for i in range(num_questions)]
    test_col = ["shuffle0-test0"] + [""] * num_questions
    rows = list(zip(prompt_col, order_col, test_col, strict=True))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _make_run_config(testing_file, output_dir, *, max_retries=3, use_structured=False):
    """Create a mock run OmegaConf-like object for example_generator."""
    run = MagicMock()
    run.testing_file = str(testing_file)
    run.output_dir = str(output_dir)
    run.model = "openai/gpt-4"
    run.name_exp = None
    run.test_count = 1
    run.shuffle_count = 0
    run.temperature = 0
    run.api_base = ""
    run.batch_size = None
    run.use_structured_output = use_structured
    run.max_parse_failure_retries = max_retries
    run.allowed_providers = ["openai", "anthropic", "gemini", "ollama"]
    return run


def _make_questionnaire(num_questions=3):
    """Create a minimal questionnaire dict."""
    return {
        "name": "Test",
        "prompt": "Rate yourself on a scale of 1-5.",
        "inner_setting": "You are a research participant.",
        "questions": {str(i + 1): f"Q{i + 1}" for i in range(num_questions)},
        "scale": 5,
        "compute_mode": "AVG",
        "reverse": [],
        "categories": [],
    }


class TestMaxParseFailureRetries:
    """Tests for the max_parse_failure_retries configuration."""

    @patch("example_generator._validate_model_litellm")
    @patch("example_generator._validate_model_provider")
    @patch("example_generator.chat")
    def test_successful_parse_no_retry(self, mock_chat, mock_validate_prov, mock_validate_lite, tmp_path):
        """When parsing succeeds on the first attempt, no retry occurs."""
        csv_path = tmp_path / "test.csv"
        _make_testing_csv(csv_path, num_questions=3)
        run = _make_run_config(csv_path, tmp_path, max_retries=3)
        questionnaire = _make_questionnaire(num_questions=3)

        mock_chat.return_value = "1: 5\n2: 3\n3: 4"

        example_generator(questionnaire, run)

        assert mock_chat.call_count == 1

    @patch("example_generator._validate_model_litellm")
    @patch("example_generator._validate_model_provider")
    @patch("example_generator.chat")
    def test_retries_on_parse_failure_then_succeeds(self, mock_chat, mock_validate_prov, mock_validate_lite, tmp_path):
        """When first attempt fails parsing but second succeeds, chat is called twice."""
        csv_path = tmp_path / "test.csv"
        _make_testing_csv(csv_path, num_questions=3)
        run = _make_run_config(csv_path, tmp_path, max_retries=3)
        questionnaire = _make_questionnaire(num_questions=3)

        # First call returns wrong number of scores (triggers length mismatch), second succeeds
        mock_chat.side_effect = [
            "1: 5\n2: 3",  # only 2 scores for 3 questions
            "1: 5\n2: 3\n3: 4",  # correct: 3 scores
        ]

        example_generator(questionnaire, run)

        assert mock_chat.call_count == 2

    @patch("example_generator._validate_model_litellm")
    @patch("example_generator._validate_model_provider")
    @patch("example_generator.chat")
    def test_exhausts_retries_and_skips(self, mock_chat, mock_validate_prov, mock_validate_lite, tmp_path):
        """When all retries are exhausted, the column is skipped (no crash)."""
        csv_path = tmp_path / "test.csv"
        _make_testing_csv(csv_path, num_questions=3)
        run = _make_run_config(csv_path, tmp_path, max_retries=2)
        questionnaire = _make_questionnaire(num_questions=3)

        # Always return wrong number of scores
        mock_chat.return_value = "1: 5"  # only 1 score for 3 questions

        # Should not raise; should log error and skip
        example_generator(questionnaire, run)

        # 1 initial attempt + 2 retries = 3 total calls
        assert mock_chat.call_count == 3

    @patch("example_generator._validate_model_litellm")
    @patch("example_generator._validate_model_provider")
    @patch("example_generator.chat")
    def test_zero_retries_fails_immediately(self, mock_chat, mock_validate_prov, mock_validate_lite, tmp_path):
        """When max_retries=0, only one attempt is made (no retries)."""
        csv_path = tmp_path / "test.csv"
        _make_testing_csv(csv_path, num_questions=3)
        run = _make_run_config(csv_path, tmp_path, max_retries=0)
        questionnaire = _make_questionnaire(num_questions=3)

        mock_chat.return_value = "1: 5"  # wrong number of scores

        example_generator(questionnaire, run)

        assert mock_chat.call_count == 1

    @patch("example_generator._validate_model_litellm")
    @patch("example_generator._validate_model_provider")
    @patch("example_generator.chat")
    def test_default_retries_is_three(self, mock_chat, mock_validate_prov, mock_validate_lite, tmp_path):
        """Default max_parse_failure_retries from config is 3 (1 + 3 = 4 total attempts)."""
        csv_path = tmp_path / "test.csv"
        _make_testing_csv(csv_path, num_questions=3)
        run = _make_run_config(csv_path, tmp_path, max_retries=3)
        questionnaire = _make_questionnaire(num_questions=3)

        mock_chat.return_value = "1: 5"  # always fails

        example_generator(questionnaire, run)

        # 1 initial + 3 retries = 4 total
        assert mock_chat.call_count == 4

    @patch("example_generator._validate_model_litellm")
    @patch("example_generator._validate_model_provider")
    @patch("example_generator.chat")
    def test_structured_output_retries(self, mock_chat, mock_validate_prov, mock_validate_lite, tmp_path):
        """Structured output mode also respects max_parse_failure_retries."""
        csv_path = tmp_path / "test.csv"
        _make_testing_csv(csv_path, num_questions=3)
        run = _make_run_config(csv_path, tmp_path, max_retries=1, use_structured=True)
        questionnaire = _make_questionnaire(num_questions=3)

        # Return only 1 answer when 3 are expected (missing q2, q3)
        bad_json = '{"q1": 5}'
        good_json = '{"q1": 5, "q2": 3, "q3": 4}'
        mock_chat.side_effect = [bad_json, good_json]

        example_generator(questionnaire, run)

        assert mock_chat.call_count == 2
