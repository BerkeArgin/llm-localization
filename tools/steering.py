from tools.contrastiveact import contrastive_act_gen_opt
from tqdm import tqdm
import pandas as pd
import torch as t


def load_steering_vec_map():
    steering_vec_map = {}
    for c_pref in ["tr", "fr", "ru", "bn", "us"]:
        
        steering_vec_map[("perculture_en", c_pref)] = t.load(f"vectors/gemma2_9b_it/per_culture/{c_pref}_en_avg_all_tasks.pt", weights_only=True)
        steering_vec_map[("perculture_trans", c_pref)] = t.load(f"vectors/gemma2_9b_it/per_culture/{c_pref}_trans_avg_all_tasks.pt", weights_only=True)
        steering_vec_map[("en_universal", c_pref)] = t.load(f"vectors/gemma2_9b_it/universal/en_universal_all_cultures.pt", weights_only=True)
        steering_vec_map[("trans_universal", c_pref)] = t.load(f"vectors/gemma2_9b_it/universal/trans_universal_all_cultures.pt", weights_only=True)
        steering_vec_map[("en_universal_loo", c_pref)] = t.load(f"vectors/gemma2_9b_it/universal/en_universal_{c_pref}_out.pt", weights_only=True)
        steering_vec_map[("transuniversal_loo", c_pref)] = t.load(f"vectors/gemma2_9b_it/universal/trans_universal_{c_pref}_out.pt", weights_only=True)
        steering_vec_map[("none", c_pref)] = t.zeros_like(steering_vec_map[("perculture_en", c_pref)])

        if c_pref != "us":
            steering_vec_map[("implicit", c_pref)] = t.load(f"vectors/gemma2_9b_it/implicit/{c_pref}_avg_all_tasks.pt", weights_only=True)

        for task in ["names", "cities", "culturedistil", "culturebench"]:
            steering_vec_map[("en_"+task, c_pref)] = t.load(f"vectors/gemma2_9b_it/per_task/{c_pref}_{task}_en.pt", weights_only=True)
            steering_vec_map[("trans_"+task, c_pref)] = t.load(f"vectors/gemma2_9b_it/per_task/{c_pref}_{task}_trans.pt", weights_only=True)
    return steering_vec_map

def run_steering_binary(
                 nnmodel,
                 tokenizer,
                 steering_vec_map, 
                 test_data, 
                 layers, 
                 alphas, 
                 batch_size,
                 countries = ["Turkey", "France", "Russia", "Bangladesh", "United States"],
                 tasks = ["names", "cities", "culturedistil", "culturebench"],
                 vector_type = "none", 
                 folder=None, filename=None):
    
    country_to_suffix= {"Turkey": "tr", "France": "fr", "Russia": "ru", "Bangladesh": "bn", "United States":"us"}
    print(f"Running {vector_type}, results will be saved to {folder}/{filename}.csv")
    
    outputs = []
    for country in countries:
        for task in tasks:
            s = country_to_suffix[country]
            if vector_type == "pertask_en":
                vec_key = ("en"+task,s)
            elif vector_type == "pertask_trans":
                vec_key = ("trans"+task,s)
            else:
                vec_key = (vector_type,s)

            if not vec_key in steering_vec_map:
                continue 
            
            print(country, task)

            steering_vec = steering_vec_map[vec_key].unsqueeze(1)
            test_entries = test_data.query("country==@country and subtask==@task").to_dict(orient="records")

            batch_entries = [test_entries[k:k+batch_size] for k in range(0, len(test_entries), batch_size)]
            batch_inputs = [[entry["input"] for entry in batch] for batch in batch_entries]

            for i,batch_imp in tqdm(enumerate(batch_inputs), total=len(batch_inputs)):
                for alpha in alphas:
                    with t.no_grad():
                        out = contrastive_act_gen_opt(nnmodel, tokenizer, alpha * steering_vec, prompt=batch_imp, layer=layers, n_new_tokens=1)
                        for j,layer in enumerate(out[0]):
                            texts = out[0][layer]
                            probs = out[1]
                            epsilon = 1e-6
                            probs[probs < epsilon] = 0

                            for k, text in enumerate(texts):
                                out_dict = {"alpha": alpha, "steer_out": text, "steer_prob": probs[j,k,:,:].to_sparse(), "layer": layer}
                                out_dict.update(batch_entries[i][k])
                                outputs.append(out_dict)
    
    pd.to_pickle(outputs, f"{folder}/{filename}.pkl")

    new_rows = []
    for out in tqdm(outputs):

        out["ans_west"] = str(int(out["ans_west_idx"]))
        out["ans_local"] = str(int(out["ans_local_idx"]))

        west_ind = tokenizer.encode(out["ans_west"], add_special_tokens=False)[0]
        out["prob_west"] = out["steer_prob"][0,west_ind].item()

        local_ind = tokenizer.encode(out["ans_local"], add_special_tokens=False)[0]
        out["prob_local"] = out["steer_prob"][0,local_ind].item()


        if out["ans_west"] in str(out["steer_out"]):
            out["ans_type"] = "west"
        elif out["ans_local"] in str(out["steer_out"]):
            out["ans_type"] = "local"
        else:
            out["ans_type"] = "none"
        new_rows.append(out)

    steer_df = pd.DataFrame(new_rows)
    steer_df.drop(columns=["steer_prob"], inplace=True)

    steer_df["vector"] = vector_type
    steer_df.to_csv(f"{folder}/{filename}.csv", index=False)
    return steer_df


def run_steering_mcqa(
                 nnmodel,
                 tokenizer,
                 steering_vec_map, 
                 test_data, 
                 layers, 
                 alphas, 
                 batch_size,
                 langs = ["tr", "fr", "ru", "bn", "en"],
                 tasks = ["names", "cities", "o1"],
                 vector_type = "none", 
                 folder=None, filename=None):
    
    lang_suffix_to_lang = {
    "tr": "Turkish",
    "fr": "French",
    "ru": "Russian",
    "bn": "Bengali",
    "en": "English",
    }
    
    print(f"Running {vector_type}, results will be saved to {folder}/{filename}.csv")
    
    outputs = []
    for lang in langs:
        for task in tasks:
            s = lang
            if lang == "en":
                s = "us"

            if vector_type == "pertask_en":
                vec_key = ("en"+task,s)
            elif vector_type == "pertask_trans":
                vec_key = ("trans"+task,s)
            else:
                vec_key = (vector_type,s)

            if not vec_key in steering_vec_map:
                continue 
            
            test_entries = test_data.query(f"lang=='{lang_suffix_to_lang[lang]}' and subtask==@task").to_dict(orient="records")

            print(lang, task)

            steering_vec = steering_vec_map[vec_key].unsqueeze(1)

            batch_entries = [test_entries[k:k+batch_size] for k in range(0, len(test_entries), batch_size)]
            batch_inputs = [[entry["input"] for entry in batch] for batch in batch_entries]

            for i,batch_imp in tqdm(enumerate(batch_inputs), total=len(batch_inputs)):
                for alpha in alphas:
                    with t.no_grad():
                        out = contrastive_act_gen_opt(nnmodel, tokenizer, alpha * steering_vec, prompt=batch_imp, layer=layers, n_new_tokens=1)
                        for j,layer in enumerate(out[0]):
                            texts = out[0][layer]
                            probs = out[1]
                            epsilon = 1e-6
                            probs[probs < epsilon] = 0

                            for k, text in enumerate(texts):
                                out_dict = {"alpha": alpha, "steer_out": text, "steer_prob": probs[j,k,:,:].to_sparse(), "layer": layer}
                                out_dict.update(batch_entries[i][k])
                                outputs.append(out_dict)
    
    pd.to_pickle(outputs, f"{folder}/{filename}.pkl")

    new_rows = []
    for out in tqdm(outputs):
        out["steer_ans_type"] = "none"
        for i in ["tr", "fr", "ru", "bn", "us"]:
            ans_idx  = str(out[f"option_{i}_idx"])
            pos = tokenizer.encode(ans_idx, add_special_tokens=False)[0]

            out["prob_"+i] = out["steer_prob"][0,pos].item()
            if ans_idx in out["steer_out"]:
                out["steer_ans_type"] = i
        new_rows.append(out)

    steer_df = pd.DataFrame(new_rows)
    steer_df.drop(columns=["steer_prob"], inplace=True)
    steer_df["vector"] = vector_type
    steer_df.to_csv(f"{folder}/{filename}.csv", index=False)

    return steer_df