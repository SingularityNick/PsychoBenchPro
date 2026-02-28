# Input tokens per questionnaire

Input = **system message** (inner_setting) + **user message** (structured prompt: questionnaire prompt + all questions + JSON instruction), as sent in the first turn of the chat flow in `example_generator`.

Tokenizers:

- **OpenAI**: `tiktoken` with encoding `cl100k_base` (same as GPT-4 / GPT-3.5), including chat template tokens.
- **Anthropic**: Claude API `messages.count_tokens` (model `claude-sonnet-4-20250514`). Requires `ANTHROPIC_API_KEY`.
- **Gemini**: Google GenAI `models.count_tokens` (model `gemini-2.0-flash`). Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- **DeepSeek**: Hugging Face `transformers` tokenizer from `deepseek-ai/DeepSeek-V2` (BPE).
- **Llama**: Hugging Face `transformers` tokenizer from `hf-internal-testing/llama-tokenizer` (Llama 2–style; for Llama 3.x use `meta-llama/Llama-3.2-1B` with `HF_TOKEN` and license acceptance).
- **Qwen**: Hugging Face `transformers` tokenizer from `Qwen/Qwen2.5-0.5B-Instruct` (BPE).

To regenerate (loads API keys from repo root `.env` if present; DeepSeek/Llama/Qwen require `transformers` and HF download):

```bash
uv run python scripts/count_questionnaire_tokens.py
```

## Input token counts (all tokenizers)

| Questionnaire | OpenAI | Anthropic | Gemini | DeepSeek | Llama | Qwen |
|---------------|--------|-----------|--------|----------|-------|------|
| Empathy       | 331    | 332       | 313    | 316      | 331   | 298  |
| BFI           | 534    | 598       | 568    | 577      | 632   | 534  |
| BSRI          | 509    | 585       | 570    | 588      | 632   | 525  |
| EPQ-R         | 1,442  | 1,569     | 1,599  | 1,603    | 1,686 | 1,500 |
| LMS           | 235    | 237       | 209    | 211      | 227   | 199  |
| DTDD          | 274    | 279       | 256    | 257      | 279   | 243  |
| ECR-R         | 720    | 746       | 761    | 764      | 797   | 713  |
| GSE           | 312    | 316       | 289    | 293      | 305   | 279  |
| ICB           | 396    | 410       | 369    | 377      | 404   | 360  |
| LOT-R         | 258    | 262       | 239    | 240      | 250   | 224  |
| EIS           | 713    | 739       | 734    | 737      | 769   | 702  |
| WLEIS         | 405    | 417       | 395    | 400      | 424   | 377  |
| CABIN         | 1,743  | 2,000     | 2,083  | 2,101    | 2,316 | 1,929 |
| 16P           | 1,028  | 1,038     | 1,044  | 1,051    | 1,111 | 1,045 |

---

## Output token upper bound (one answer per questionnaire)

For **structured JSON output** with **worst-case pretty-printing** (indent=2, one key per line, **one digit per value**), the following are **upper bounds** for the number of **output** tokens per single response, by tokenizer.

| Questionnaire | N   | OpenAI | DeepSeek | Llama | Qwen |
|---------------|-----|--------|----------|-------|------|
| Empathy       | 10  | 82     | 93       | 93    | 83   |
| BFI           | 44  | 354    | 433      | 433   | 389  |
| BSRI          | 60  | 482    | 593      | 593   | 533  |
| EPQ-R         | 100 | 802    | 994      | 994   | 894  |
| LMS           | 9   | 74     | 83       | 83    | 74   |
| DTDD          | 12  | 98     | 113      | 113   | 101  |
| ECR-R         | 36  | 290    | 353      | 353   | 317  |
| GSE           | 10  | 82     | 93       | 93    | 83   |
| ICB           | 8   | 66     | 74       | 74    | 66   |
| LOT-R         | 10  | 82     | 93       | 93    | 83   |
| EIS           | 33  | 266    | 323     | 323   | 290  |
| WLEIS         | 16  | 130    | 153     | 153   | 137  |
| CABIN         | 164 | 1,314  | 1,698   | 1,698 | 1,534 |
| 16P           | 60  | 482    | 593     | 593   | 533  |

Assumptions: response is a single JSON object `{"q1": 1, "q2": 5, ...}`, formatted with 2-space indent and newline after each key-value pair; each value is a single digit (1–9). Anthropic and Gemini output token counts are comparable to OpenAI (same order of magnitude). Compact JSON would use fewer tokens.

---

## Total token estimates (all questionnaires × power-analysis run counts)

Based on [Power analysis: BFI model vs. crowd](POWER_ANALYSIS_BFI_CROWD.md): required repeat runs **n₁** (per model, per questionnaire) for α = 0.01, power 80%, for different effect sizes **d** and crowd sizes **n₂**. Each row assumes that **the same number of runs is applied to every questionnaire** (not just BFI), so totals are for one model across all 14 questionnaires.

Token basis (OpenAI tiktoken): **8,900 input** and **4,604 output** per full pass (one run on each of the 14 questionnaires). Totals below = **R × 8,900** input + **R × 4,604** output.

| Runs (n₁) | Scenario (d, n₂) | Input tokens | Output tokens | Total tokens |
|-----------|------------------|--------------|---------------|--------------|
| 77        | d = 0.8, n₂ = 25 | 685,300      | 354,508       | 1,039,808    |
| 31        | d = 0.8, n₂ = 50 | 275,900      | 142,724       | 418,624      |
| 26        | d = 1.0, n₂ = 25 | 231,400      | 119,704       | 351,104      |
| 17        | d = 1.0, n₂ = 50 | 151,300      | 78,268        | 229,568      |
| 14        | d = 1.2, n₂ = 25 | 124,600      | 64,456        | 189,056      |
| 11        | d = 1.2, n₂ = 50 | 97,900       | 50,644        | 148,544      |
| 10        | current (underpowered) | 89,000  | 46,040        | 135,040      |

Example: for **77 runs** per questionnaire (recommended for medium–large effects with 25 models in the crowd), one model over all 14 questionnaires uses **~1.04M tokens** (~685K input, ~355K output). **Total cost for 77 runs × all models** in `conf/benchmark.yaml`: see [BENCHMARK_COST_77_RUNS.md](BENCHMARK_COST_77_RUNS.md).
