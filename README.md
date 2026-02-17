<div align= "center">
    <h1> 🔍🤖PsychoBench</h1>
</div>

<div align="center">
    <img src="https://codecov.io/gh/SingularityNick/PsychoBenchPro/branch/main/graph/badge.svg" alt="Coverage" />
</div>

</div>

<div align="center">
<img src="framework.jpg" width="750px">
</div>

**RESEARCH USE ONLY✅ NO COMMERCIAL USE ALLOWED❌**

Benchmarking LLMs' Psychological Portray.

**UPDATES**

[Jan 16 2024]: PsychoBench is accepted to **ICLR 2024 Oral (1.2%)**

[Dec 28 2023]: Add support to 16personalities.com

## Setup

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

1. **Install uv** (if needed): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Install dependencies**: `uv sync` (creates `.venv` and installs from `pyproject.toml` and `uv.lock`)
3. **Run** either with `uv run python run_psychobench.py ...` or by activating the venv: `source .venv/bin/activate` then `python run_psychobench.py ...`

## 🛠️ Usage
✨An example run (with uv: use `uv run python run_psychobench.py ...` or activate `.venv` first):
```
python run_psychobench.py model=gpt-3.5-turbo questionnaire=EPQ-R openai_key="<openai_api_key>" shuffle_count=1 test_count=2
```

## Configuration
Configuration is driven by [Hydra](https://hydra.cc/). Defaults live in `conf/config.yaml`. You can override any option from the command line (e.g. `model=gpt-4`, `questionnaire=BFI,EPQ-R`). The `openai_key` option defaults to the `OPENAI_API_KEY` environment variable if set; you can also set it in the config file or override it on the CLI. Outputs are written under `results/` (Hydra’s run directory is set to `.` so the working directory does not change).

✨An example result:
| Category | gpt-4 (n = 10) | Male (n = 693) | Female (n = 878) |
| :---: | :---: | :---: | :---: |
| Extraversion | 13.9 $\pm$ 4.3 | 12.5 $\pm$ 6.0 | 14.1 $\pm$ 5.1 | 
| Pschoticism | 17.8 $\pm$ 2.1 | 7.2 $\pm$ 4.6 | 5.7 $\pm$ 3.9 | 
| Neuroticism | 3.9 $\pm$ 6.0 | 10.5 $\pm$ 5.8 | 12.5 $\pm$ 5.1 | 
| Lying | 7.0 $\pm$ 2.1 | 7.1 $\pm$ 4.3 | 6.9 $\pm$ 4.0 | 

## 🧪 Tests
Unit tests use **pytest** and cover core logic in `utils.py` and `example_generator.py` (e.g. `get_questionnaire`, `convert_data`, `compute_statistics`, `hypothesis_testing`, `parsing`, `convert_results`).

**Install dependencies** (managed by uv):
```bash
uv sync --dev
```

**Run all tests (from the project root):**
```bash
uv run pytest tests/ -v
```

**Run with coverage:**
```bash
uv run pytest tests/ -v --cov=utils --cov=example_generator
```

## 🔧 Config options and CLI overrides
All options are defined in `conf/config.yaml` and can be overridden from the command line (e.g. `model=gpt-4`, `questionnaire=BFI`).

1. **questionnaire** (required): Select the questionnaire(s) to run. See the list below. Example: `questionnaire=EPQ-R` or `questionnaire=BFI,DTDD,EPQ-R`.

2. **model**: The name of the model to test (e.g. `text-davinci-003`, `gpt-3.5-turbo`, `gpt-4`).

3. **shuffle_count**: Number of question orders. 0 = original only; n > 0 = original plus n permutations. Default: 0.

4. **test_count**: Number of runs per order. Default: 1.

5. **name_exp**: Name of this run (used for result filenames). Default: use model name.

6. **significance_level**: Significance level for hypothesis testing (human vs LLM means). Default: 0.01.

7. **mode**: Pipeline stage: `auto` (full), `generation`, `testing`, or `analysis`. Default: `auto`.

8. **openai_key**: OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable if set; override via config or CLI when using the example generator.

## 🦙 Benchmarking Your Own Model
It is easy! Just replace the function `example_generator` fed into the function `run_psychobench(cfg, generator)`.

Your customized function `your_generator(questionnaire, run)` receives the current questionnaire and a **run config** (Hydra-style config with per-questionnaire paths). It should:

1. Read questions from the file `run.testing_file`. That file lives under `results/` (see `run_psychobench()` in `utils.py`) and has the following format:

| Prompt: ... | order-1 | shuffle0-test0 | shuffle0-test1 | Prompt: ... | order-2 | shuffle0-test0 | shuffle0-test1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 1 | | | Q3 | 3 | | |
| Q2 | 2 | | | Q5 | 5 | | |
| ... | ... | | | ... | ... | | |
| Qn | n | | | Q1 | 1 | | |

You can read the columns before each column starting with `order-`, which contains the shuffled questions for your input.

2. Call your own LLM and get the results.

3. Fill in the blank in the file `run.testing_file`. **Remember**: No need to map the response to its original order. Our code will take care of it.

Please check `example_generator.py` for datailed information.

## 📃 Questionnaire List (config key: questionnaire)
To include **multiple** questionnaires, use a comma-separated override: `questionnaire=BFI,DTDD,EPQ-R`.

To include **ALL** questionnaires: `questionnaire=ALL`.

1. Big Five Inventory: `questionnaire=BFI`
2. Dark Triad Dirty Dozen: `questionnaire=DTDD`
3. Eysenck Personality Questionnaire-Revised: `questionnaire=EPQ-R`
4. Experiences in Close Relationships-Revised (Adult Attachment Questionnaire): `questionnaire=ECR-R`
5. Comprehensive Assessment of Basic Interests: `questionnaire=CABIN`
6. General Self-Efficacy: `questionnaire=GSE`
7. Love of Money Scale: `questionnaire=LMS`
8. Bem's Sex Role Inventory: `questionnaire=BSRI`
9. Implicit Culture Belief: `questionnaire=ICB`
10. Revised Life Orientation Test: `questionnaire=LOT-R`
11. Empathy Scale: `questionnaire=Empathy`
12. Emotional Intelligence Scale: `questionnaire=EIS`
13. Wong and Law Emotional Intelligence Scale: `questionnaire=WLEIS`

## 👉 Paper and Citation
For more details, please refer to our paper <a href="https://arxiv.org/abs/2310.01386">here</a>.

If you find our paper&tool interesting and useful, please feel free to give us a star and cite us through:
```
@inproceedings{huang2024humanity,
  title={On the humanity of conversational ai: Evaluating the psychological portrayal of llms},
  author={Huang, Jen-tse and Wang, Wenxuan and Li, Eric John and Lam, Man Ho and Ren, Shujie and Yuan, Youliang and Jiao, Wenxiang and Tu, Zhaopeng and Lyu, Michael},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
