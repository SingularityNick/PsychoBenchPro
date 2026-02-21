# Pricing reference for benchmark models

How pricing is calculated and where to find official rates for each model in `conf/benchmark.yaml`.

## How pricing is calculated

All listed providers bill **per token** (or per million tokens), separately for **input** and **output**:

- **Formula:**  
  `cost = (input_tokens / 1_000_000) × input_$ per_1M + (output_tokens / 1_000_000) × output_$ per_1M`

- **Input tokens:** prompt (system + user messages, plus any tool definitions). For chat, this includes the provider’s chat template (e.g. `<|im_start|>system` …).

- **Output tokens:** completion (model reply). For structured JSON output, this is the generated JSON only.

- **Units:** Prices are almost always quoted **per 1M tokens** ($/MTok or $/1M). Convert to per-token by dividing by 1,000,000.

- **No markup (OpenRouter):** OpenRouter does not mark up provider pricing; you pay the provider’s rate. See [OpenRouter FAQ](https://openrouter.ai/docs/faq): “We do not mark up provider pricing.”

- **Caching / batch:** Anthropic, OpenAI, and Google offer prompt caching and/or batch APIs with different (often lower) rates; the formula above uses **standard on-demand** input/output rates unless noted.

---

## Official documentation

| Provider    | Official pricing page |
|------------|------------------------|
| **Anthropic** | [docs.anthropic.com – Pricing](https://docs.anthropic.com/en/docs/about-claude/pricing) |
| **OpenAI**    | [openai.com/api/pricing](https://openai.com/api/pricing), [platform.openai.com – Pricing](https://platform.openai.com/docs/pricing) |
| **Google (Gemini)** | [Gemini API – Pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| **OpenRouter** | Per-model in [OpenRouter Models](https://openrouter.ai/models); [Pricing FAQ](https://openrouter.ai/docs/faq) (no markup) |

---

## Models in `conf/benchmark.yaml` and pricing (standard rates)

Rates below are **USD per 1M tokens** (input / output) from the official docs or OpenRouter model pages as of the dates cited. Check the links for current values.

### Anthropic (Claude API)

| Model (config id) | Input ($/1M) | Output ($/1M) | Source |
|-------------------|--------------|---------------|--------|
| `anthropic/claude-sonnet-4-20250514` | 3 | 15 | [Anthropic Pricing](https://docs.anthropic.com/en/docs/about-claude/pricing) – Claude Sonnet 4 |
| `anthropic/claude-sonnet-4-5-20250929` | 3 | 15 | Same – Claude Sonnet 4.5 |
| `anthropic/claude-opus-4-6` | 5 | 25 | Same – Claude Opus 4.6 |
| `anthropic/claude-sonnet-4-6` | 3 | 15 | Same – Claude Sonnet 4.6 |

**Calculation:** Input and output are billed separately; cache reads/writes and long-context (>200K) have different multipliers (see doc).

---

### Google (Gemini API)

| Model (config id) | Input ($/1M) | Output ($/1M) | Source |
|-------------------|--------------|---------------|--------|
| `gemini/gemini-2.5-pro` | 1.25 | 10 | [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing); >200K context: 2.50 / 15 |
| `gemini/gemini-3-pro-preview` | 2 | 12 | Same; >200K: 4 / 18; batch: 1 / 6 (≤200K) |
| `gemini/gemini-3.1-pro-preview` | 2 | 12 | Same as Gemini 3 Pro Preview |

**Calculation:** Same formula; standard tier unless using batch or long-context.

---

### OpenAI (Chat Completions / API)

| Model (config id) | Input ($/1M) | Output ($/1M) | Source |
|-------------------|--------------|---------------|--------|
| `openai/gpt-3.5-turbo-0125` | (legacy) | (legacy) | [OpenAI Pricing](https://openai.com/api/pricing), [Platform Pricing](https://platform.openai.com/docs/pricing) – “Legacy” / text tokens table |
| `openai/gpt-4` | (see doc) | (see doc) | Same – GPT-4 tier |
| `openai/gpt-4.1-2025-04-14` | (see doc) | (see doc) | Same – GPT-4.1 / fine-tuning table (inference may differ) |
| `openai/gpt-4o-2024-11-20` | (see doc) | (see doc) | Same – GPT-4o tier |
| `openai/gpt-5-2025-08-07` | (see doc) | (see doc) | Same – GPT-5 tier |
| `openai/gpt-5.2-2025-12-11` | 1.75 | 14 | [OpenAI Pricing](https://openai.com/api/pricing) – GPT-5.2 |
| `openai/gpt-5.2-pro-2025-12-11` | 21 | 168 | Same – GPT-5.2 Pro |
| `openai/text-davinci-003` | (legacy) | (legacy) | [Platform Pricing](https://platform.openai.com/docs/pricing) – “Legacy models” |

**Calculation:** Same per-token formula; reasoning/output tokens are billed as output. Batch API gives ~50% off; cached input has lower rate (see doc).

---

### OpenRouter (DeepSeek, Moonshot, Meta Llama, Qwen)

OpenRouter uses **provider pricing** (no markup). Each model has its own page under `openrouter.ai/<provider>/<model>` with **Input Price** and **Output Price** per 1M tokens (or per 1K in the UI; convert to per 1M for the formula).

| Model (config id) | Input ($/1M) | Output ($/1M) | Source |
|-------------------|--------------|---------------|--------|
| `openrouter/deepseek/deepseek-v3` | (see model page) | (see model page) | [OpenRouter – DeepSeek](https://openrouter.ai/models?q=deepseek) |
| `openrouter/deepseek/deepseek-v3.2` | (see model page) | (see model page) | Same |
| `openrouter/moonshotai/kimi-k2` | (see model page) | (see model page) | [OpenRouter – Moonshot](https://openrouter.ai/models?q=moonshot) |
| `openrouter/meta-llama/llama-3.1-405b-instruct` | 4 | 4 | [OpenRouter – Llama 3.1 405B](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct) |
| `openrouter/meta-llama/llama-3.1-70b-instruct` | (see model page) | (see model page) | [OpenRouter – Llama](https://openrouter.ai/models?q=llama) |
| `openrouter/meta-llama/llama-4-maverick-17b-128e-instruct` | ~0.18 | ~0.60 | OpenRouter model page (example; check current) |
| `openrouter/meta-llama/llama-2-13b` | (see model page) | (see model page) | Same |
| `openrouter/meta-llama/llama-2-7b` | (see model page) | (see model page) | Same |
| `openrouter/qwen/qwen2.5-72b-instruct` | (see model page) | (see model page) | [OpenRouter – Qwen](https://openrouter.ai/models?q=qwen) |
| `openrouter/qwen/qwen3-235b-a22b-instruct` | (see model page) | (see model page) | Same |
| `openrouter/qwen/qwen3-32b` | (see model page) | (see model page) | Same |
| `openrouter/qwen/qwen3-coder-480b-a35b-instruct` | (see model page) | (see model page) | Same |

**Calculation:** Same formula; use the **Input Price** and **Output Price** from the model’s OpenRouter page (convert to $/1M if shown per 1K). Cache read/write pricing is per-model if offered.

---

## Using this with token counts

- **Input tokens per run:** See [QUESTIONNAIRE_TOKEN_COUNTS.md](QUESTIONNAIRE_TOKEN_COUNTS.md) for per-questionnaire and per–full-pass (all 14 questionnaires) input/output token counts by tokenizer.
- **Cost per run (one model, one full pass):**  
  `cost = (input_tokens / 1e6) * input_$ per_1M + (output_tokens / 1e6) * output_$ per_1M`  
  using that model’s tokenizer and the provider’s $/1M for that model.
- **Cost for R runs (e.g. power-analysis scenarios):** Multiply the per-run token totals by R, then apply the formula above (or multiply per-run cost by R if you already computed cost per run).

Tokenizer differences (OpenAI vs Anthropic vs Gemini vs DeepSeek/Llama/Qwen) mean the same prompt can have different token counts; use the tokenizer that matches the model when estimating cost.
