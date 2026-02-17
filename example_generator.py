import os
import re
import time

import litellm
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

# Models that use prompt-based completion (legacy OpenAI text models).
# All other models (GPT chat, Claude, Gemini, etc.) use chat with messages.
TEXT_COMPLETION_MODEL_PATTERN = re.compile(
    r"^(openai/)?(text-(davinci|curie|babbage|ada)-\d{3})(\b|$)",
    re.IGNORECASE,
)


def _is_text_completion_model(model: str) -> bool:
    """Return True if model uses prompt-based completion (not chat messages)."""
    return bool(TEXT_COMPLETION_MODEL_PATTERN.search(model))


def _provider_prefix(model: str) -> str | None:
    """Return the provider prefix from a model string (e.g. 'openai' from 'openai/gpt-4'), or None."""
    m = (model or "").strip()
    if "/" in m:
        return m.split("/", 1)[0].lower()
    return None


def _provider_from_model(model: str) -> str:
    """Map provider prefix to API key attribute: gemini -> google, anthropic -> anthropic, openai -> openai."""
    prefix = _provider_prefix(model)
    if prefix is None:
        return "openai"  # fallback only when validation is skipped
    if prefix == "gemini":
        return "google"
    if prefix in ("anthropic", "openai"):
        return prefix
    return "openai"


def _validate_model_provider(model: str, allowed_providers: list) -> None:
    """Raise ValueError if model does not start with one of the allowed provider prefixes."""
    if not model or not (model := model.strip()):
        raise ValueError("model must be set and non-empty")
    prefix = _provider_prefix(model)
    if prefix is None:
        raise ValueError(
            f"model must start with a provider prefix, e.g. openai/, anthropic/, gemini/. "
            f"Allowed providers: {allowed_providers}. Got: {model!r}"
        )
    allowed = [p.lower() for p in (allowed_providers or [])]
    if prefix not in allowed:
        raise ValueError(
            f"model provider {prefix!r} is not in allowed_providers {allowed}. "
            f"Model must start with one of: {', '.join(p + '/' for p in allowed)}"
        )


def _llm_completion(
    model: str,
    *,
    messages=None,
    prompt=None,
    temperature=0,
    n=1,
    max_tokens=1024,
    api_key=None,
    api_base=None,
    delay=1,
):
    """Call LiteLLM completion. Supports any provider (OpenAI, Anthropic, Gemini, etc.)."""
    time.sleep(delay)
    kwargs = {
        "model": model,
        "temperature": temperature,
        "n": n,
        "max_tokens": max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base
    if messages is not None:
        kwargs["messages"] = messages
    elif prompt is not None:
        kwargs["prompt"] = prompt
    else:
        raise ValueError("Either messages or prompt must be provided")

    kwargs["drop_params"] = True  # drop unsupported params (e.g. temperature on O-series)
    response = litellm.completion(**kwargs)
    # Chat models return message.content; text completion models return .text
    if n == 1:
        choice = response.choices[0]
        if hasattr(choice, "message") and choice.message is not None:
            return choice.message.content or ""
        return getattr(choice, "text", "") or ""
    choices = response.choices
    choices.sort(key=lambda x: x.index)
    return [
        (c.message.content if hasattr(c, "message") and c.message else None)
        or getattr(c, "text", "")
        for c in choices
    ]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat(
    model,
    messages,
    temperature=0,
    n=1,
    max_tokens=1024,
    api_key=None,
    api_base=None,
    delay=1,
):
    """Chat completion for any LiteLLM-supported model (GPT, Claude, Gemini, etc.)."""
    return _llm_completion(
        model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base,
        delay=delay,
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion(
    model,
    prompt,
    temperature=0,
    n=1,
    max_tokens=1024,
    api_key=None,
    api_base=None,
    delay=1,
):
    """Text completion for legacy models (text-davinci-003, etc.)."""
    return _llm_completion(
        model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base,
        delay=delay,
    )


def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except Exception:
        result_list = ["" for element in result.split('\n')]
        logger.warning("Unable to capture the responses on {}.", column_header)

    return result_list


def example_generator(questionnaire, run):
    testing_file = run.testing_file
    model = run.model
    records_file = run.name_exp if run.name_exp is not None else model

    allowed = getattr(run, "allowed_providers", None) or ["gemini", "anthropic", "openai"]
    _validate_model_provider(model, allowed)

    api_base = getattr(run, "api_base", None) or ""

    # Read the existing CSV file into a pandas DataFrame
    df = pd.read_csv(testing_file)

    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * run.test_count

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                questions_column_index = i - 1
                shuffle_count += 1

                # Retrieve the column data as a string
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [
                    questions_list[i:i+30] for i in range(0, len(questions_list), 30)
                ]
                questions_list = [
                    '\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)])
                    for j, questions in enumerate(separated_questions)
                ]


                for k in range(run.test_count):

                    df = pd.read_csv(testing_file)

                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'

                    while(True):
                        result_string_list = []
                        previous_records = []

                        for questions_string in questions_list:
                            result = ""
                            use_text_completion = _is_text_completion_model(model)
                            api_kw = {"api_base": api_base} if api_base else {}

                            if use_text_completion:
                                inner_setting = questionnaire["inner_setting"].replace(
                                    'Format: "index: score"', 'Format: "index: score\\\n"'
                                )
                                inputs = inner_setting + questionnaire["prompt"] + "\n" + questions_string
                                result = completion(model, inputs, **api_kw)
                            else:
                                # Chat models: GPT, Claude, Gemini, Llama, etc.
                                inputs = previous_records + [
                                    {"role": "system", "content": questionnaire["inner_setting"]},
                                    {"role": "user", "content": questionnaire["prompt"] + "\n" + questions_string},
                                ]
                                result = chat(model, inputs, **api_kw)
                                previous_records.append({
                                    "role": "user",
                                    "content": questionnaire["prompt"] + "\n" + questions_string,
                                })
                                previous_records.append({"role": "assistant", "content": result})

                            result_string_list.append(result.strip())

                            # Write the prompts and results to the run-specific output dir
                            prompts_dir = os.path.join(run.output_dir, "prompts")
                            responses_dir = os.path.join(run.output_dir, "responses")
                            os.makedirs(prompts_dir, exist_ok=True)
                            os.makedirs(responses_dir, exist_ok=True)

                            prompts_path = os.path.join(
                                prompts_dir,
                                f'{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt',
                            )
                            with open(prompts_path, "a") as file:
                                file.write(f'{inputs}\n====\n')
                            responses_path = os.path.join(
                                responses_dir,
                                f'{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt',
                            )
                            with open(responses_path, "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)

                        result_list = convert_results(result_string, column_header)

                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except Exception:
                            logger.warning("Unable to capture the responses on {}.", column_header)

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)

                    pbar.update(1)
