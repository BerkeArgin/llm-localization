from tools.globals import DEMONSTRATIONS
from tools.prompts import format_base, format_multi_choice, format_v2, format_multi_choice_5choice
import pandas as pd
import re

def change_numbers(prompt):
    # Match the options in the form of "1. Option1\n2. Option2"
    pattern = r"\n1\.\s*(.*?)\n2\.\s*(.*?)(?=\n|$)"

    def change_match(match):
        option1, option2 = match.groups()
        return f"\nA. {option1}\nB. {option2}"

    # Substitute the matched pattern with swapped options
    swapped_prompt = re.sub(pattern, change_match, prompt)
    swapped_prompt = swapped_prompt.replace("1,2", "A,B")

    return swapped_prompt
def base_chat_template(messages):
    str_to_return = ""
    for m in messages:
        str_to_return += f"{m['content']}\n"
    return str_to_return

def messages_to_str(messages, tokenizer, instruction_model=False):
    if type(messages) == str:
        messages = [{"role":"user", "content":messages}]
    if instruction_model:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return base_chat_template(messages)

def get_demonstration(formatter, lang):
    demos = []
    for demo in DEMONSTRATIONS[lang]:
        demos.append({"role":"user", "content":formatter(demo)})
        demos.append({"role":"assistant", "content":demo["answer"]})
    return demos

def prepare_dataset_base(data_df, tokenizer=None):
    data_df["prompt"] = data_df.apply(format_base, axis=1)
    data_df["messages"] = data_df.apply(lambda x: get_demonstration(format_base,x["lang"])+[{"role": "user", "content": x["prompt"]}], axis=1)
    if tokenizer:
        to_input = lambda x: messages_to_str(x, tokenizer, instruction_model=False)
        data_df["input"] = data_df["messages"].apply(to_input)
    return data_df

def prepare_dataset_it(data_df, tokenizer=None):
    data_df["prompt"] = data_df.apply(format_multi_choice, axis=1)
    data_df["messages"] = data_df.apply(lambda x: [{"role": "user", "content": x["prompt"]}], axis=1)
    if tokenizer:
        to_input = lambda x: messages_to_str(x, tokenizer, instruction_model=True)
        data_df["input"] = data_df["messages"].apply(to_input)
    return data_df

def prepare_dataset_steer(data_df, tokenizer=None):
    prompt_suffix = {
        "English": "My guess is **",
        "Turkish": "Tahminim **",
        "French": "Ma supposition est **",
        "Russian": "Моё предположение **",
        "Bengali": "আমার অনুমান হলো **",
    }

    data_df["prompt"] = data_df.apply(format_v2, axis=1)
    data_df["messages"] = data_df.apply(lambda x: [{"role": "user", "content": x["prompt"]}], axis=1)
    if tokenizer:
        to_input = lambda x: messages_to_str(x, tokenizer, instruction_model=True)
        data_df["input"] = data_df["messages"].apply(to_input)
        data_df["input"] += data_df["lang"].apply(lambda x: f"{prompt_suffix[x]}")
    return data_df

def prepare_dataset_5choice(data_df, tokenizer=None):
    data_df["prompt"] = data_df.apply(format_multi_choice_5choice, axis=1)
    data_df["messages"] = data_df.apply(lambda x: [{"role": "user", "content": x["prompt"]}], axis=1)
    if tokenizer:
        to_input = lambda x: messages_to_str(x, tokenizer, instruction_model=True)
        data_df["input"] = data_df["messages"].apply(to_input)
    return data_df