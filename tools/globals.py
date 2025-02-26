import json
import os

def load_lang_headers():
    lang_headers_path = os.path.join(os.path.dirname(__file__),"json_data/lang_headers.json")
    with open(lang_headers_path,"r") as f:
        lang_headers = json.load(f)
    lang_header_map = {}
    for lang in lang_headers:
        ent = lang_headers[lang]
        lang_header_map[lang] = (ent["question"], ent["option"], ent["answer"])
    return lang_header_map

def load_demonstrations():
    demonstration_path = os.path.join(os.path.dirname(__file__),"json_data/icl_demonstrations.json")
    with open(demonstration_path,"r") as f:
        demonstrations = json.load(f)
    return demonstrations

def load_prompt_lang_map():
    prompt_lang_map_path = os.path.join(os.path.dirname(__file__),"json_data/eval_instructions.json")
    
    with open(prompt_lang_map_path,"r") as f:
        prompt_lang_list = json.load(f)

    prompt_lang_map = {}

    for ent in prompt_lang_list:
        lang = ent["lang"]
        prompt_lang_map[lang] = ent["prompt"]
    
    return prompt_lang_map


def reload_globals():
    global DEMONSTRATIONS, PROMPT_LANG_MAP, LANG_HEADER_MAP, COUNTRY_HEADER_MAP, COUNTRY_LANGUAGE_MAP
    DEMONSTRATIONS = load_demonstrations()
    PROMPT_LANG_MAP = load_prompt_lang_map()
    LANG_HEADER_MAP = load_lang_headers()

DEMONSTRATIONS = load_demonstrations()
PROMPT_LANG_MAP = load_prompt_lang_map()
LANG_HEADER_MAP = load_lang_headers()
DEFAULT_HEADER = ("Question:","Options:","Answer:")


