"""Unit tests for example_generator multi-provider LLM support."""
import sys

sys.path.insert(0, ".")  # noqa: E402

from example_generator import _is_text_completion_model


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
