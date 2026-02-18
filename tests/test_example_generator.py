"""Unit tests for example_generator multi-provider LLM support and structured output."""
import csv
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")  # noqa: E402

from example_generator import (
    QuestionnaireResponse,
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
# QuestionnaireResponse Pydantic model
# ---------------------------------------------------------------------------


class TestQuestionnaireResponse:
    """Tests for the Pydantic structured output model."""

    def test_valid_json_parses(self):
        raw = (
            '{"answers": [{"question_index": "1", "score": 5},'
            ' {"question_index": "2", "score": 3},'
            ' {"question_index": "3", "score": 7}]}'
        )
        resp = QuestionnaireResponse.model_validate_json(raw)
        assert len(resp.answers) == 3
        assert resp.answers[0].question_index == "1"
        assert resp.answers[0].score == 5
        assert resp.answers[2].question_index == "3"
        assert resp.answers[2].score == 7

    def test_empty_answers_list(self):
        raw = '{"answers": []}'
        resp = QuestionnaireResponse.model_validate_json(raw)
        assert resp.answers == []

    def test_missing_answers_key_raises(self):
        raw = '{"scores": [1, 2, 3]}'
        with pytest.raises(ValueError):
            QuestionnaireResponse.model_validate_json(raw)

    def test_missing_required_field_raises(self):
        raw = '{"answers": [{"question_index": "1"}]}'
        with pytest.raises(ValueError):
            QuestionnaireResponse.model_validate_json(raw)

    def test_non_integer_score_raises(self):
        raw = '{"answers": [{"question_index": "1", "score": "high"}]}'
        with pytest.raises(ValueError):
            QuestionnaireResponse.model_validate_json(raw)

    def test_schema_generation(self):
        """The model can generate a JSON schema (used by LiteLLM for structured output)."""
        schema = QuestionnaireResponse.model_json_schema()
        assert "answers" in schema.get("properties", {})
        assert schema["properties"]["answers"]["type"] == "array"
        items = schema["properties"]["answers"].get("items", {})
        if "$ref" in items:
            defs = schema.get("$defs", {})
            ref_name = items["$ref"].split("/")[-1]
            item_schema = defs.get(ref_name, {})
        else:
            item_schema = items
        assert "required" in item_schema
        assert "question_index" in item_schema["required"]
        assert "score" in item_schema["required"]


# ---------------------------------------------------------------------------
# convert_results_structured
# ---------------------------------------------------------------------------


class TestConvertResultsStructured:
    """Tests for the structured JSON response parser."""

    def test_valid_new_format_returns_scores(self):
        raw = (
            '{"answers": [{"question_index": "1", "score": 5},'
            ' {"question_index": "2", "score": 3},'
            ' {"question_index": "3", "score": 7}]}'
        )
        result = convert_results_structured(raw, "test-col")
        assert result == [5, 3, 7]

    def test_empty_answers_returns_empty(self):
        raw = '{"answers": []}'
        result = convert_results_structured(raw, "test-col")
        assert result == []

    def test_json_with_extra_keys_still_parsed(self):
        """Response may include $defs or other schema keys; parser uses only 'answers'."""
        raw = (
            '{"$defs": {"AnswerItem": {}},'
            ' "answers": [{"question_index": "1", "score": 4},'
            ' {"question_index": "2", "score": 2}]}'
        )
        result = convert_results_structured(raw, "test-col")
        assert result == [4, 2]

    def test_invalid_json_falls_back_to_text_parser(self):
        """When JSON parsing fails, falls back to legacy text parser."""
        text = "1: 5\n2: 3\n3: 7"
        result = convert_results_structured(text, "test-col")
        assert result == [5, 3, 7]

    def test_malformed_json_falls_back(self):
        """Malformed JSON (missing quotes) triggers fallback."""
        text = "{answers: bad}"
        result = convert_results_structured(text, "test-col")
        assert isinstance(result, list)

    def test_scores_are_integers(self):
        raw = (
            '{"answers": [{"question_index": "1", "score": 1},'
            ' {"question_index": "2", "score": 7},'
            ' {"question_index": "3", "score": 4}]}'
        )
        result = convert_results_structured(raw, "test-col")
        assert all(isinstance(s, int) for s in result)


# ---------------------------------------------------------------------------
# _build_structured_prompt
# ---------------------------------------------------------------------------


class TestBuildStructuredPrompt:
    """Tests for the structured output prompt builder."""

    def test_includes_questions(self):
        q = {"prompt": "Rate these:", "inner_setting": "You are a helper."}
        questions = "1. I am talkative.\n2. I am reserved."
        prompt = _build_structured_prompt(q, questions)
        assert "1. I am talkative." in prompt
        assert "2. I am reserved." in prompt

    def test_includes_json_format_instructions(self):
        q = {"prompt": "Rate these:", "inner_setting": "You are a helper."}
        prompt = _build_structured_prompt(q, "1. Q1")
        assert "JSON" in prompt
        assert '"answers"' in prompt
        assert "question_index" in prompt
        assert "score" in prompt
        assert '{"question_index": "1", "score": 5}' in prompt

    def test_includes_base_prompt(self):
        q = {"prompt": "Please evaluate yourself.", "inner_setting": "You are a helper."}
        prompt = _build_structured_prompt(q, "1. Q1")
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

        # Return only 1 answer when 3 are expected
        bad_json = '{"answers": [{"question_index": "1", "score": 5}]}'
        good_json = (
            '{"answers": [{"question_index": "1", "score": 5},'
            ' {"question_index": "2", "score": 3},'
            ' {"question_index": "3", "score": 4}]}'
        )
        mock_chat.side_effect = [bad_json, good_json]

        example_generator(questionnaire, run)

        assert mock_chat.call_count == 2
