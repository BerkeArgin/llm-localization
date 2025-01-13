from tqdm import trange
import torch as t
import matplotlib.pyplot as plt
import numpy as np

def patch_scope_lens(nnmodel, tokenizer, model_output, verbose=False, target_prompt=None, token_index=-1):
    
    if target_prompt is None:
        id_prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
    else:
        id_prompt_target = target_prompt

    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    all_logits = []

    residuals = model_output["residuals"]

    lrange = trange(len(nnmodel.model.layers)) if verbose else range(len(nnmodel.model.layers))
    for i in lrange:
        with nnmodel.trace(id_prompt_tokens.repeat(residuals.shape[1], 1), validate=False, scan=False):
            nnmodel.model.layers[i].output[0][:,token_index,:] = residuals[i, :, :]
            logits = nnmodel.lm_head.output[:,token_index, :].save()
        all_logits.append(logits.value.detach().cpu())

    all_logits = t.stack(all_logits)

    return all_logits


def patch_scope_gen(nnmodel, tokenizer, model_output, verbose=False,
                    target_prompt=None, target_token_idx=-1,
                    n_new_tokens=10):
    """
    residuals: (n_layers, batch_size, seq_len, dmodel)
    returns a list of completions when patching at different layers, and the token probabilites
    """
    residuals = model_output["residuals"]
    if residuals.shape[1] > 1:
        raise ValueError("not implemented for batch size > 1")
    if target_prompt is None:
        id_prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
    else:
        id_prompt_target = target_prompt
    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    probas = []
    lrange = trange(len(nnmodel.model.layers)) if verbose else range(len(nnmodel.model.layers))
    l2toks = {}
    for i in lrange:
        toks = id_prompt_tokens.repeat(residuals.shape[1], 1)
        start_len = toks.shape[1]
        probas_tok = []
        for idx_tok in range(n_new_tokens):
            T = toks.shape[1]
            offset = -idx_tok if target_token_idx<0 else 0
            with nnmodel.trace(toks, validate=False, scan=False):
                nnmodel.model.layers[i].output[0][:, target_token_idx+offset,:] = residuals[i, :, -1, :]
                logits = nnmodel.lm_head.output[:, -1, :].save()
            probas_tok.append(logits.value.softmax(dim=-1).detach().cpu())
            pred_tok = t.argmax(logits.value, dim=-1, keepdim=True)
            toks = t.cat([toks, pred_tok.to(toks.device)], dim=-1)
            l2toks[i] = toks.detach().cpu()[:, start_len:]
        probas.append(t.stack(probas_tok))
    probas = t.stack(probas)
    probas = probas[:, :, 0]
    # go from toks to strings
    return {k: [tokenizer.decode(t) for t in v] for k, v in l2toks.items()}, probas

def plot_probs(data_row, probs, token_index=0, tokenizer=None):
    ans_west = data_row["ans_west"]
    ans_local = data_row["ans_local"]
    ind_west = tokenizer.encode(ans_west, add_special_tokens=False)[0]
    ind_local = tokenizer.encode(ans_local, add_special_tokens=False)[0]

    west_probs = probs[:, token_index, ind_west].float().numpy()
    local_probs = probs[:, token_index, ind_local].float().numpy()


    plt.figure(figsize=(10, 6))
    x = np.arange(west_probs.shape[0])
    plt.scatter(x, west_probs, color='blue', s=10)
    plt.scatter(x, local_probs, color='orange', s=10)
    plt.plot(west_probs, label="West", color='blue')
    plt.plot(local_probs, label="Local", color='orange')
    plt.grid(True)
    plt.legend()


def plot_avg_probs_ax(data_df, results, tokenizer, token_index=0, ax=None):
    west_total = []
    local_total = []
    
    for _, data_row in data_df.iterrows():
        ans_west = data_row["ans_west"]
        ans_local = data_row["ans_local"]
        question_id = data_row["question_id"]
        
        ind_west = tokenizer.encode(ans_west, add_special_tokens=False)[0]
        ind_local = tokenizer.encode(ans_local, add_special_tokens=False)[0]
        
        probs = results[question_id][1]
        
        west_probs_across_layers = probs[:, token_index, ind_west].float().numpy()
        local_probs_across_layers = probs[:, token_index, ind_local].float().numpy()
        
        west_total.append(west_probs_across_layers)
        local_total.append(local_probs_across_layers)
    
    west_total = np.array(west_total)
    local_total = np.array(local_total)
    
    west_mean = west_total.mean(axis=0)
    local_mean = local_total.mean(axis=0)
    west_std = west_total.std(axis=0, ddof=1)   # ddof=1 for unbiased estimate
    local_std = local_total.std(axis=0, ddof=1)

    # Number of questions
    N = west_total.shape[0]
    
    # Standard error of the mean
    west_sem = west_std / np.sqrt(N)
    local_sem = local_std / np.sqrt(N)

    # 95% CI = mean +/- z * SEM (for normal distribution, z ~ 1.96)
    z_val = 1.96
    west_ci = z_val * west_sem
    local_ci = z_val * local_sem
    
    x = np.arange(west_mean.shape[0])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x, west_mean, color='blue')
    ax.scatter(x, local_mean, color='orange')
    ax.plot(x, west_mean, color='blue', label='Non Loc. Prob')
    ax.fill_between(x, west_mean - west_ci, west_mean + west_ci, alpha=0.2, color='blue')
    ax.plot(x, local_mean, color='orange', label='Loc. Prob')
    ax.fill_between(x, local_mean - local_ci, local_mean + local_ci, alpha=0.2, color='orange')
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Probability')
    ax.set_title('Average Token Probability Across Layers')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if ax is None:
        plt.show()
def plot_avg_probs(data_df, results, tokenizer, token_index=0, ax=None):
    """
    For each row in data_df, this function extracts the probabilities
    for 'ans_west' and 'ans_local' across different layers (patchscope results).
    Then it computes the mean and standard deviation across all items (rows).
    Finally, it plots the mean probability with a shaded region for the std dev.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        A DataFrame where each row contains at least "question_id", "ans_west", "ans_local".
    results : dict
        A dictionary keyed by question_id, containing patchscope results. Example structure:
        results[question_id] = (some_data, probs_array)
        where probs_array likely has shape (num_layers, ..., vocab_size).
    tokenizer : PreTrainedTokenizer
        A tokenizer used to encode the answer strings.
    token_index : int
        Index along the sequence dimension (if needed). Defaults to 0.
    """
    
    west_total = []
    local_total = []
    
    # Collect probabilities for each row
    for _, data_row in data_df.iterrows():
        ans_west = data_row["ans_west"]
        ans_local = data_row["ans_local"]
        question_id = data_row["question_id"]
        
        # Encode tokens
        ind_west = tokenizer.encode(ans_west, add_special_tokens=False)[0]
        ind_local = tokenizer.encode(ans_local, add_special_tokens=False)[0]
        
        # Retrieve probability array for this question
        # Suppose results[question_id] = (patchscope_info, probs)
        # and `probs` is shaped [num_layers, seq_length, vocab_size]
        probs = results[question_id][1]
        
        # Extract the probabilities across layers for the single token_index
        # We get shape: (num_layers,)
        west_probs_across_layers = probs[:, token_index, ind_west].float().numpy()
        local_probs_across_layers = probs[:, token_index, ind_local].float().numpy()
        
        west_total.append(west_probs_across_layers)
        local_total.append(local_probs_across_layers)
    
    # Convert list of arrays to a single 2D array: shape (num_questions, num_layers)
    west_total = np.array(west_total)  # shape (N, L)
    local_total = np.array(local_total)  # shape (N, L)


    
    # Compute mean and standard deviation across questions (axis=0)
    west_mean = west_total.mean(axis=0)
    west_std = west_total.std(axis=0)
    local_mean = local_total.mean(axis=0)
    local_std = local_total.std(axis=0)
     # Compute standard deviation across questions (axis=0)
    west_std = west_total.std(axis=0, ddof=1)   # ddof=1 for unbiased estimate
    local_std = local_total.std(axis=0, ddof=1)

    # Number of questions
    N = west_total.shape[0]
    
    # Standard error of the mean
    west_sem = west_std / np.sqrt(N)
    local_sem = local_std / np.sqrt(N)

    # 95% CI = mean +/- z * SEM (for normal distribution, z ~ 1.96)
    z_val = 1.96
    west_ci = z_val * west_sem
    local_ci = z_val * local_sem
    # Create x-axis for the layers
    x = np.arange(west_mean.shape[0])
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # West line + confidence interval
    plt.scatter(x, west_mean, color='blue')
    plt.scatter(x, local_mean, color='orange')
    plt.plot(x, west_mean, color='blue', label='Non Loc. Prob')
    plt.fill_between(x, west_mean - west_ci, west_mean + west_ci, alpha=0.2, color='blue')
    
    # Local line + confidence interval
    plt.plot(x, local_mean, color='orange', label='Loc. Prob')
    plt.fill_between(x, local_mean - local_ci, local_mean + local_ci, alpha=0.2, color='orange')
    
    # Labels, legend, grid
    plt.xlabel('Layer Index')
    plt.ylabel('Probability')
    plt.title('Average Token Probability Across Layers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_localization_rate(data_df, results, tokenizer, token_index=0, ax=None):
    """
    For each row in data_df, this function extracts the probabilities
    for 'ans_west' and 'ans_local' across different layers (patchscope results).
    Then it computes the mean and standard deviation across all items (rows).
    Finally, it plots the mean probability with a shaded region for the std dev.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        A DataFrame where each row contains at least "question_id", "ans_west", "ans_local".
    results : dict
        A dictionary keyed by question_id, containing patchscope results. Example structure:
        results[question_id] = (some_data, probs_array)
        where probs_array likely has shape (num_layers, ..., vocab_size).
    tokenizer : PreTrainedTokenizer
        A tokenizer used to encode the answer strings.
    token_index : int
        Index along the sequence dimension (if needed). Defaults to 0.
    """
    
    west_total = []
    local_total = []
    
    # Collect probabilities for each row
    for _, data_row in data_df.iterrows():
        ans_west = data_row["ans_west"]
        ans_local = data_row["ans_local"]
        question_id = data_row["question_id"]
        
        # Encode tokens
        ind_west = tokenizer.encode(ans_west, add_special_tokens=False)[0]
        ind_local = tokenizer.encode(ans_local, add_special_tokens=False)[0]
        
        # Retrieve probability array for this question
        # Suppose results[question_id] = (patchscope_info, probs)
        # and `probs` is shaped [num_layers, seq_length, vocab_size]
        probs = results[question_id][1]
        
        
        # Extract the probabilities across layers for the single token_index
        # We get shape: (num_layers,)
        next_tokens = probs[:, token_index, :].argmax(dim=-1)
        west_probs_across_layers = (next_tokens == ind_west).float().numpy()
        local_probs_across_layers = (next_tokens == ind_local).float().numpy()
        
        west_total.append(west_probs_across_layers)
        local_total.append(local_probs_across_layers)
    
    # Convert list of arrays to a single 2D array: shape (num_questions, num_layers)
    west_total = np.array(west_total)  # shape (N, L)
    local_total = np.array(local_total)  # shape (N, L)


    
    # Compute mean and standard deviation across questions (axis=0)
    west_mean = west_total.sum(axis=0)
    local_mean = local_total.sum(axis=0)
    
    # Create x-axis for the layers
    x = np.arange(west_mean.shape[0])
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # West line + confidence interval
    plt.scatter(x, west_mean, color='blue', s=10)
    plt.scatter(x, local_mean, color='orange', s=10)
    plt.plot(x, west_mean, color='blue', label='Non Loc. Prob')
    
    # Local line + confidence interval
    plt.plot(x, local_mean, color='orange', label='Loc. Prob')
    
    # Labels, legend, grid
    plt.xlabel('Layer Index')
    plt.ylabel('Probability')
    plt.title('Average Token Probability Across Layers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()