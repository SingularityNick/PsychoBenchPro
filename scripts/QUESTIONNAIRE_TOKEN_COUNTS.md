# Input tokens per questionnaire

Input = **system message** (inner_setting) + **user message** (structured prompt: questionnaire prompt + all questions + JSON instruction), as sent in the first turn of the chat flow in `example_generator`.

Tokenizers:

- **OpenAI**: `tiktoken` with encoding `cl100k_base` (same as GPT-4 / GPT-3.5), including chat template tokens.
- **Anthropic**: Claude API `messages.count_tokens` (model `claude-sonnet-4-20250514`). Requires `ANTHROPIC_API_KEY`.
- **Gemini**: Google GenAI `models.count_tokens` (model `gemini-2.0-flash`). Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

To regenerate (loads API keys from repo root `.env` if present):

```bash
uv run python scripts/count_questionnaire_tokens.py
```

## Counts (OpenAI / Anthropic / Gemini)

| Questionnaire | OpenAI (tiktoken) | Anthropic | Gemini |
|---------------|------------------|-----------|--------|
| Empathy       | 331              | 332       | 313    |
| BFI           | 534              | 598       | 568    |
| BSRI          | 509              | 585       | 570    |
| EPQ-R         | 1,442            | 1,569     | 1,599  |
| LMS           | 235              | 237       | 209    |
| DTDD          | 274              | 279       | 256    |
| ECR-R         | 720              | 746       | 761    |
| GSE           | 312              | 316       | 289    |
| ICB           | 396              | 410       | 369    |
| LOT-R         | 258              | 262       | 239    |
| EIS           | 713              | 739       | 734    |
| WLEIS         | 405              | 417       | 395    |
| CABIN         | 1,743            | 2,000     | 2,083  |
| 16P           | 1,028            | 1,038     | 1,044  |

---

## Output token upper bound (one answer per questionnaire)

For **structured JSON output** with **worst-case pretty-printing** (indent=2, one key per line, **one digit per value**), the following are reasonable **upper bounds** for the number of **output** tokens per single response (OpenAI tiktoken `cl100k_base`; other providers are typically within the same order of magnitude).

| Questionnaire | # Questions | Output tokens (upper bound) |
|---------------|-------------|-----------------------------|
| Empathy       | 10          | 82                          |
| BFI           | 44          | 354                         |
| BSRI          | 60          | 482                         |
| EPQ-R         | 100         | 802                         |
| LMS           | 9           | 74                          |
| DTDD          | 12          | 98                          |
| ECR-R         | 36          | 290                         |
| GSE           | 10          | 82                          |
| ICB           | 8           | 66                          |
| LOT-R         | 10          | 82                          |
| EIS           | 33          | 266                         |
| WLEIS         | 16          | 130                         |
| CABIN         | 164         | 1,314                       |
| 16P           | 60          | 482                         |

Assumptions: response is a single JSON object `{"q1": 1, "q2": 5, ...}`, formatted with 2-space indent and newline after each key-value pair; each value is a single digit (1–9). Compact JSON would use fewer tokens.
