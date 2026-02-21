# Total cost: 77 runs × all questionnaires × all benchmark models

**Scenario:** Run the benchmark with **77 shuffle runs** per model (power-analysis recommendation for d ≈ 0.8, n₂ = 25). Each run is one full pass over all 14 questionnaires. Token counts are tokenizer-matched; pricing is standard on-demand from [PRICING_REFERENCE.md](PRICING_REFERENCE.md) and official provider docs.

**Formula:** `cost = (input_tokens / 1e6) × input_$ per_1M + (output_tokens / 1e6) × output_$ per_1M`

**Total (27 models):** **$186.91 USD**

---

## Per-model cost (77 runs each)

| Model | Tokenizer | Input tokens | Output tokens | Cost (USD) |
|------|-----------|--------------|---------------|------------|
| anthropic/claude-sonnet-4-20250514 | Anthropic | 733,656 | 354,508 | 7.52 |
| anthropic/claude-sonnet-4-5-20250929 | Anthropic | 733,656 | 354,508 | 7.52 |
| anthropic/claude-opus-4-6 | Anthropic | 733,656 | 354,508 | 12.53 |
| anthropic/claude-sonnet-4-6 | Anthropic | 733,656 | 354,508 | 7.52 |
| gemini/gemini-2.5-pro | Gemini | 726,033 | 354,508 | 4.45 |
| gemini/gemini-3-pro-preview | Gemini | 726,033 | 354,508 | 5.71 |
| gemini/gemini-3.1-pro-preview | Gemini | 726,033 | 354,508 | 5.71 |
| openai/gpt-3.5-turbo-0125 | OpenAI | 685,300 | 354,508 | 0.87 |
| openai/gpt-4 | OpenAI | 685,300 | 354,508 | 17.49 |
| openai/gpt-4.1-2025-04-14 | OpenAI | 685,300 | 354,508 | 6.31 |
| openai/gpt-4o-2024-11-20 | OpenAI | 685,300 | 354,508 | 10.52 |
| openai/gpt-5-2025-08-07 | OpenAI | 685,300 | 354,508 | 4.40 |
| openai/gpt-5.2-2025-12-11 | OpenAI | 685,300 | 354,508 | 6.16 |
| openai/gpt-5.2-pro-2025-12-11 | OpenAI | 685,300 | 354,508 | 73.95 |
| openai/text-davinci-003 | OpenAI | 685,300 | 354,508 | 2.08 |
| openrouter/deepseek/deepseek-v3 | DeepSeek | 732,655 | 438,053 | 0.63 |
| openrouter/deepseek/deepseek-v3.2 | DeepSeek | 732,655 | 438,053 | 0.63 |
| openrouter/moonshotai/kimi-k2 | OpenAI (proxy) | 685,300 | 354,508 | 3.15 |
| openrouter/meta-llama/llama-3.1-405b-instruct | Llama | 782,551 | 438,053 | 4.88 |
| openrouter/meta-llama/llama-3.1-70b-instruct | Llama | 782,551 | 438,053 | 1.07 |
| openrouter/meta-llama/llama-4-maverick-17b-128e-instruct | Llama | 782,551 | 438,053 | 0.40 |
| openrouter/meta-llama/llama-2-13b | Llama | 782,551 | 438,053 | 0.22 |
| openrouter/meta-llama/llama-2-7b | Llama | 782,551 | 438,053 | 0.07 |
| openrouter/qwen/qwen2.5-72b-instruct | Qwen | 687,456 | 394,009 | 0.62 |
| openrouter/qwen/qwen3-235b-a22b-instruct | Qwen | 687,456 | 394,009 | 0.97 |
| openrouter/qwen/qwen3-32b | Qwen | 687,456 | 394,009 | 0.22 |
| openrouter/qwen/qwen3-coder-480b-a35b-instruct | Qwen | 687,456 | 394,009 | 1.30 |

---

## Regenerate

```bash
uv run python scripts/benchmark_cost_77runs.py
```

Pricing and tokenizer mapping are defined in the script; update them when [PRICING_REFERENCE.md](PRICING_REFERENCE.md) or [QUESTIONNAIRE_TOKEN_COUNTS.md](QUESTIONNAIRE_TOKEN_COUNTS.md) change. Some OpenRouter $/1M values are from model pages or estimates—check [OpenRouter models](https://openrouter.ai/models) for current rates.
