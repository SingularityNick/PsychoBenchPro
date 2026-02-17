import os
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


def _configure_litellm(run):
    """Set up LiteLLM credentials from the run config.

    Resolution order for the API key:
      1. ``run.api_key`` (generic, works with any provider)
      2. ``run.openai_key`` (legacy fallback for backward compatibility)
      3. Provider-specific env vars already present (OPENAI_API_KEY,
         ANTHROPIC_API_KEY, …) — LiteLLM picks these up automatically.
    """
    api_key = getattr(run, "api_key", None) or ""
    openai_key = getattr(run, "openai_key", None) or ""

    effective_key = api_key or openai_key
    if effective_key:
        litellm.api_key = effective_key
        os.environ.setdefault("OPENAI_API_KEY", effective_key)

    api_base = getattr(run, "api_base", None)
    if api_base:
        litellm.api_base = api_base


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat(
    model,                      # Any LiteLLM-supported model identifier
    messages,                   # [{"role": "system"/"user"/"assistant", "content": "Hello!"}]
    temperature=0,    # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,                        # Chat completion choices to generate for each input message.
    max_tokens=1024,            # The maximum number of tokens to generate in the chat completion.
    delay=1,          # Seconds to sleep after each request.
    api_base=None,    # Optional custom API base URL.
):
    time.sleep(delay)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "n": n,
        "max_tokens": max_tokens,
    }
    if api_base:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    if n == 1:
        return response.choices[0].message.content
    else:
        return [i.message.content for i in response.choices]


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
    api_base = getattr(run, "api_base", None) or None

    _configure_litellm(run)

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
                            inputs = previous_records + [
                                {"role": "system", "content": questionnaire["inner_setting"]},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string},
                            ]
                            result = chat(model, inputs, api_base=api_base)
                            previous_records.append({
                                "role": "user",
                                "content": questionnaire["prompt"] + '\n' + questions_string,
                            })
                            previous_records.append({"role": "assistant", "content": result})

                            result_string_list.append(result.strip())

                            # Write the prompts and results to the file
                            os.makedirs("prompts", exist_ok=True)
                            os.makedirs("responses", exist_ok=True)

                            prompts_path = (
                                f'prompts/{records_file}-{questionnaire["name"]}'
                                f'-shuffle{shuffle_count - 1}.txt'
                            )
                            with open(prompts_path, "a") as file:
                                file.write(f'{inputs}\n====\n')
                            responses_path = (
                                f'responses/{records_file}-{questionnaire["name"]}'
                                f'-shuffle{shuffle_count - 1}.txt'
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
