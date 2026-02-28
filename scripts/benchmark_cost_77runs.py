#!/usr/bin/env python3
"""
Compute total cost for 77 shuffle runs per model (all questionnaires) in conf/benchmark.yaml.

Uses token totals from QUESTIONNAIRE_TOKEN_COUNTS.md and $/1M from PRICING_REFERENCE.md.
Cost = (input_tokens/1e6)*input_$perM + (output_tokens/1e6)*output_$perM.
"""

from __future__ import annotations

# Per full pass (14 questionnaires), from QUESTIONNAIRE_TOKEN_COUNTS.md
INPUT_SUM = {
    "OpenAI": 331 + 534 + 509 + 1442 + 235 + 274 + 720 + 312 + 396 + 258 + 713 + 405 + 1743 + 1028,
    "Anthropic": 332 + 598 + 585 + 1569 + 237 + 279 + 746 + 316 + 410 + 262 + 739 + 417 + 2000 + 1038,
    "Gemini": 313 + 568 + 570 + 1599 + 209 + 256 + 761 + 289 + 369 + 239 + 734 + 395 + 2083 + 1044,
    "DeepSeek": 316 + 577 + 588 + 1603 + 211 + 257 + 764 + 293 + 377 + 240 + 737 + 400 + 2101 + 1051,
    "Llama": 331 + 632 + 632 + 1686 + 227 + 279 + 797 + 305 + 404 + 250 + 769 + 424 + 2316 + 1111,
    "Qwen": 298 + 534 + 525 + 1500 + 199 + 243 + 713 + 279 + 360 + 224 + 702 + 377 + 1929 + 1045,
}
OUTPUT_SUM = {
    "OpenAI": 82 + 354 + 482 + 802 + 74 + 98 + 290 + 82 + 66 + 82 + 266 + 130 + 1314 + 482,
    "Anthropic": None,  # use OpenAI as proxy
    "Gemini": None,
    "DeepSeek": 93 + 433 + 593 + 994 + 83 + 113 + 353 + 93 + 74 + 93 + 323 + 153 + 1698 + 593,
    "Llama": 93 + 433 + 593 + 994 + 83 + 113 + 353 + 93 + 74 + 93 + 323 + 153 + 1698 + 593,
    "Qwen": 83 + 389 + 533 + 894 + 74 + 101 + 317 + 83 + 66 + 83 + 290 + 137 + 1534 + 533,
}
OUTPUT_SUM["Anthropic"] = OUTPUT_SUM["OpenAI"]
OUTPUT_SUM["Gemini"] = OUTPUT_SUM["OpenAI"]

# Models from conf/benchmark.yaml: (tokenizer, input $/1M, output $/1M)
# Sources: PRICING_REFERENCE.md + official docs; OpenRouter from model pages / search
MODELS = [
    # Anthropic
    ("anthropic/claude-sonnet-4-20250514", "Anthropic", 3, 15),
    ("anthropic/claude-sonnet-4-5-20250929", "Anthropic", 3, 15),
    ("anthropic/claude-opus-4-6", "Anthropic", 5, 25),
    ("anthropic/claude-sonnet-4-6", "Anthropic", 3, 15),
    # Gemini
    ("gemini/gemini-2.5-pro", "Gemini", 1.25, 10),
    ("gemini/gemini-3-pro-preview", "Gemini", 2, 12),
    ("gemini/gemini-3.1-pro-preview", "Gemini", 2, 12),
    # OpenAI
    ("openai/gpt-3.5-turbo-0125", "OpenAI", 0.50, 1.50),
    ("openai/gpt-4", "OpenAI", 10, 30),
    ("openai/gpt-4.1-2025-04-14", "OpenAI", 3, 12),
    ("openai/gpt-4o-2024-11-20", "OpenAI", 5, 20),
    ("openai/gpt-5-2025-08-07", "OpenAI", 1.25, 10),
    ("openai/gpt-5.2-2025-12-11", "OpenAI", 1.75, 14),
    ("openai/gpt-5.2-pro-2025-12-11", "OpenAI", 21, 168),
    ("openai/text-davinci-003", "OpenAI", 2, 2),
    # OpenRouter
    ("openrouter/deepseek/deepseek-v3", "DeepSeek", 0.34, 0.88),
    ("openrouter/deepseek/deepseek-v3.2", "DeepSeek", 0.34, 0.88),
    ("openrouter/moonshotai/kimi-k2", "OpenAI", 1.50, 6.00),  # proxy tokenizer; typical OpenRouter tier
    ("openrouter/meta-llama/llama-3.1-405b-instruct", "Llama", 4, 4),
    ("openrouter/meta-llama/llama-3.1-70b-instruct", "Llama", 0.88, 0.88),
    ("openrouter/meta-llama/llama-4-maverick-17b-128e-instruct", "Llama", 0.18, 0.60),
    ("openrouter/meta-llama/llama-2-13b", "Llama", 0.18, 0.18),
    ("openrouter/meta-llama/llama-2-7b", "Llama", 0.06, 0.06),
    ("openrouter/qwen/qwen2.5-72b-instruct", "Qwen", 0.54, 0.62),
    ("openrouter/qwen/qwen3-235b-a22b-instruct", "Qwen", 0.90, 0.90),
    ("openrouter/qwen/qwen3-32b", "Qwen", 0.20, 0.20),
    ("openrouter/qwen/qwen3-coder-480b-a35b-instruct", "Qwen", 1.20, 1.20),
]


def main() -> None:
    R = 77
    total_cost = 0.0
    rows = []
    for model_id, tok, pin, pout in MODELS:
        in_sum = INPUT_SUM[tok]
        out_sum = OUTPUT_SUM[tok]
        in_tok = R * in_sum
        out_tok = R * out_sum
        cost = (in_tok / 1e6) * pin + (out_tok / 1e6) * pout
        total_cost += cost
        rows.append((model_id, tok, in_tok, out_tok, pin, pout, cost))

    # Print
    print("77 runs (all 14 questionnaires) per model — standard on-demand pricing")
    print("=" * 100)
    print(f"{'Model':<52} {'Tokenizer':<10} {'Input tok':>12} {'Output tok':>12} {'Cost (USD)':>12}")
    print("-" * 100)
    for model_id, tok, in_tok, out_tok, _pin, _pout, cost in rows:
        print(f"{model_id:<52} {tok:<10} {in_tok:>12,} {out_tok:>12,} {cost:>12.2f}")
    print("-" * 100)
    print(f"{'TOTAL (all 27 models)':<52} {'':<10} {'':<12} {'':<12} {total_cost:>12.2f}")
    print("=" * 100)
    print("\nPricing from PRICING_REFERENCE.md and official provider docs; OpenRouter from model pages.")
    print("Token counts from QUESTIONNAIRE_TOKEN_COUNTS.md (tokenizer-matched).")


if __name__ == "__main__":
    main()
