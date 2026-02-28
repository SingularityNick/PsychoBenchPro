#!/usr/bin/env python3
"""
Count input/output tokens per questionnaire using multiple tokenizers.

Tokenizers: OpenAI (tiktoken), Anthropic, Gemini, DeepSeek, Llama, Qwen (Hugging Face).
Builds the same prompt structure as example_generator (system + structured user message).
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


# --- Hugging Face (DeepSeek, Llama, Qwen) ---
_HF_TOKENIZER_CACHE: dict[str, object] = {}


def _get_hf_tokenizer(model_id: str):
    """Load and cache Hugging Face tokenizer by model id."""
    if model_id not in _HF_TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _HF_TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _HF_TOKENIZER_CACHE[model_id]


def _count_hf_tokens(model_id: str, text: str) -> int:
    """Return token count for text using a Hugging Face tokenizer."""
    tok = _get_hf_tokenizer(model_id)
    enc = tok.encode(text, add_special_tokens=False)
    return len(enc)


def count_deepseek_tokens(system: str, user: str) -> int | None:
    """Input token count using DeepSeek tokenizer (same text as Gemini: system + user)."""
    try:
        full = f"{system}\n\n{user}"
        return _count_hf_tokens("deepseek-ai/DeepSeek-V2", full)
    except Exception:
        return None


def count_llama_tokens(system: str, user: str) -> int | None:
    """Input token count using Llama tokenizer (same text as Gemini: system + user)."""
    try:
        full = f"{system}\n\n{user}"
        # Use public tokenizer; for Llama 3.x use meta-llama/Llama-3.2-1B with HF_TOKEN + license
        return _count_hf_tokens("hf-internal-testing/llama-tokenizer", full)
    except Exception:
        return None


def count_qwen_tokens(system: str, user: str) -> int | None:
    """Input token count using Qwen tokenizer (same text as Gemini: system + user)."""
    try:
        full = f"{system}\n\n{user}"
        return _count_hf_tokens("Qwen/Qwen2.5-0.5B-Instruct", full)
    except Exception:
        return None


def count_output_tokens_hf(model_id: str, json_str: str) -> int | None:
    """Output token count for worst-case JSON using a Hugging Face tokenizer."""
    try:
        return _count_hf_tokens(model_id, json_str)
    except Exception:
        return None


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
        try:
            deepseek_n = count_deepseek_tokens(system, user)
            deepseek_n = deepseek_n if deepseek_n is not None else "-"
        except Exception as e:
            deepseek_n = f"err: {e!r}"
        try:
            llama_n = count_llama_tokens(system, user)
            llama_n = llama_n if llama_n is not None else "-"
        except Exception as e:
            llama_n = f"err: {e!r}"
        try:
            qwen_n = count_qwen_tokens(system, user)
            qwen_n = qwen_n if qwen_n is not None else "-"
        except Exception as e:
            qwen_n = f"err: {e!r}"
        rows.append((name, openai_n, anthropic_n, gemini_n, deepseek_n, llama_n, qwen_n))

    # Print input table (6 tokenizers)
    w = 10
    headers = ("Questionnaire", "OpenAI", "Anthropic", "Gemini", "DeepSeek", "Llama", "Qwen")
    total_w = 12 + 6 * (w + 1)
    print("Questionnaire input tokens (system + structured user message)")
    print("=" * total_w)
    print(
        f"{headers[0]:<12} {headers[1]:>{w}} {headers[2]:>{w}} {headers[3]:>{w}} "
        f"{headers[4]:>{w}} {headers[5]:>{w}} {headers[6]:>{w}}"
    )
    print("-" * total_w)
    for row in rows:
        print(
            f"{row[0]:<12} {str(row[1]):>{w}} {str(row[2]):>{w}} {str(row[3]):>{w}} "
            f"{str(row[4]):>{w}} {str(row[5]):>{w}} {str(row[6]):>{w}}"
        )
    print("=" * total_w)
    if any(r[2] == "-" or r[3] == "-" for r in rows):
        print("\nSet ANTHROPIC_API_KEY and GEMINI_API_KEY (or GOOGLE_API_KEY) to include those columns.")
    if any(r[4] == "-" or r[5] == "-" or r[6] == "-" for r in rows):
        print("DeepSeek/Llama/Qwen require transformers and HF model download (may need HF_TOKEN for gated models).")

    # Output token upper bound (structured JSON, worst-case pretty-printed)
    out_w = 8
    h = ("Questionnaire", "N", "OpenAI", "DeepSeek", "Llama", "Qwen")
    out_total_w = 12 + 5 * (out_w + 1)
    print("\nOutput token upper bound (structured JSON, indent=2, 1-digit values)")
    print("-" * out_total_w)
    print(f"{h[0]:<12} {h[1]:>{out_w}} {h[2]:>{out_w}} {h[3]:>{out_w}} {h[4]:>{out_w}} {h[5]:>{out_w}}")
    print("-" * out_total_w)
    for q in questionnaires:
        n = len(q["questions"])
        json_str = worst_case_json_output(n)
        o_openai = count_output_tokens_openai(json_str)
        o_ds = count_output_tokens_hf("deepseek-ai/DeepSeek-V2", json_str)
        o_llama = count_output_tokens_hf("hf-internal-testing/llama-tokenizer", json_str)
        o_qwen = count_output_tokens_hf("Qwen/Qwen2.5-0.5B-Instruct", json_str)
        o_ds = o_ds if o_ds is not None else "-"
        o_llama = o_llama if o_llama is not None else "-"
        o_qwen = o_qwen if o_qwen is not None else "-"
        print(
            f"{q['name']:<12} {n:>{out_w}} {o_openai:>{out_w}} {str(o_ds):>{out_w}} "
            f"{str(o_llama):>{out_w}} {str(o_qwen):>{out_w}}"
        )
    print("-" * out_total_w)


if __name__ == "__main__":
    main()
