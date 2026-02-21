#!/usr/bin/env python3
"""
Count input tokens per questionnaire using Gemini, OpenAI, and Anthropic tokenizers.

Builds the same prompt structure as example_generator (system + structured user message)
and reports token counts from each provider's tokenizer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env() -> None:
    """Load .env from repo root so API keys are available."""
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")


def _build_questions_string(questionnaire: dict) -> str:
    """Build questions string from questionnaire (same order as JSON keys, numeric)."""
    qs = questionnaire["questions"]
    order = sorted(qs.keys(), key=int)
    return "\n".join(f"{i}. {qs[i]}" for i in order)


def _build_structured_prompt(questionnaire: dict, questions_string: str, n_questions: int) -> str:
    """Match example_generator's structured prompt."""
    base_prompt = questionnaire["prompt"]
    return (
        f"{base_prompt}\n{questions_string}\n\n"
        f"Respond with a JSON object with keys q1 through q{n_questions}, "
        "where each value is your integer rating (1-5)."
    )


def build_inputs(questionnaire: dict) -> tuple[str, str]:
    """Return (system_content, user_content) for chat input."""
    questions_string = _build_questions_string(questionnaire)
    n_questions = len(questionnaire["questions"])
    user_content = _build_structured_prompt(questionnaire, questions_string, n_questions)
    system_content = questionnaire["inner_setting"]
    return system_content, user_content


def worst_case_json_output(n_questions: int, *, max_value: int = 9) -> str:
    """Worst-case pretty-printed JSON: q1..qN with 1-digit values and indent=2."""
    obj = {f"q{i}": max_value for i in range(1, n_questions + 1)}
    return json.dumps(obj, indent=2)


def count_output_tokens_openai(json_str: str) -> int:
    """Count tokens in a string using OpenAI's cl100k_base (same as GPT-4 output)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(json_str))


# --- OpenAI (tiktoken) ---
def _openai_chat_format(system: str, user: str) -> str:
    """Format as OpenAI chat messages for token counting (cl100k_base)."""
    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def count_openai_tokens(system: str, user: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = _openai_chat_format(system, user)
    return len(enc.encode(text))


# --- Anthropic ---
def count_anthropic_tokens(system: str, user: str) -> int | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.count_tokens(
        model="claude-sonnet-4-20250514",
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.input_tokens


# --- Gemini ---
def count_gemini_tokens(system: str, user: str) -> int | None:
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        return None
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    full_text = f"{system}\n\n{user}"
    result = client.models.count_tokens(model="gemini-2.0-flash", contents=full_text)
    return result.total_tokens


def main() -> None:
    _load_env()
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "questionnaires.json"
    with open(path) as f:
        questionnaires = json.load(f)

    rows = []
    for q in questionnaires:
        name = q["name"]
        system, user = build_inputs(q)
        try:
            openai_n = count_openai_tokens(system, user)
        except Exception as e:
            openai_n = f"err: {e!r}"
        try:
            anthropic_n = count_anthropic_tokens(system, user)
            if anthropic_n is None:
                anthropic_n = "-"
        except Exception as e:
            anthropic_n = f"err: {e!r}"
        try:
            gemini_n = count_gemini_tokens(system, user)
            if gemini_n is None:
                gemini_n = "-"
        except Exception as e:
            gemini_n = f"err: {e!r}"
        rows.append((name, openai_n, anthropic_n, gemini_n))

    # Print table
    w = 14
    print("Questionnaire input tokens (system + structured user message)")
    print("=" * (12 + 3 * (w + 1)))
    print(f"{'Questionnaire':<12} {'OpenAI':>{w}} {'Anthropic':>{w}} {'Gemini':>{w}}")
    print("-" * (12 + 3 * (w + 1)))
    for name, o, a, g in rows:
        print(f"{name:<12} {str(o):>{w}} {str(a):>{w}} {str(g):>{w}}")
    print("=" * (12 + 3 * (w + 1)))
    if any(r[2] == "-" or r[3] == "-" for r in rows):
        print("\nSet ANTHROPIC_API_KEY and GEMINI_API_KEY (or GOOGLE_API_KEY) to include those columns.")

    # Output token upper bound (structured JSON, worst-case pretty-printed)
    print("\nOutput token upper bound (structured JSON, indent=2, 1-digit values)")
    print("(OpenAI tiktoken cl100k_base — reasonable upper bound for all providers)\n")
    out_w = 10
    print(f"{'Questionnaire':<12} {'Questions':>{out_w}} {'Output tokens':>{out_w}}")
    print("-" * (12 + 2 * (out_w + 1)))
    for q in questionnaires:
        n = len(q["questions"])
        json_str = worst_case_json_output(n)
        out_tokens = count_output_tokens_openai(json_str)
        print(f"{q['name']:<12} {n:>{out_w}} {out_tokens:>{out_w}}")
    print("-" * (12 + 2 * (out_w + 1)))


if __name__ == "__main__":
    main()
