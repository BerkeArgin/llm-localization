import matplotlib.colors as mcolors
from tqdm import trange
import pandas as pd
import torch as t
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict
from tqdm.auto import trange
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer


def contrastive_act_lens(nnmodel, tokenizer, intervene_vec, intervene_tok=-1,target_prompt = None, verbose=False):
    if target_prompt is None:
        id_prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
    else:
        id_prompt_target = target_prompt

    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    all_logits = []
    lrange = trange(len(nnmodel.model.layers)) if verbose else range(len(nnmodel.model.layers))
    for i in lrange:
        with nnmodel.trace(id_prompt_tokens.repeat(intervene_vec.shape[1], 1), validate=False, scan=False):
            nnmodel.model.layers[i].output[0][:,intervene_tok,:] += intervene_vec[i, :, :]
            logits = nnmodel.lm_head.output[:, -1, :].save()
        all_logits.append(logits.value.detach().cpu())
        
    all_logits = t.stack(all_logits)
    return all_logits


def contrastive_act_gen(nnmodel, tokenizer, intervene_vec, intervene_tok=-1, verbose=False,
                        prompt=None, n_new_tokens=10, layer=None):
    """
    residuals: (n_layers, batch_size, seq_len, dmodel)
    returns a list of completions when patching at different layers, and the token probabilities
    """

    prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    probas = []
    layers = range(len(nnmodel.model.layers)) if layer is None else layer
    lrange = trange(len(layers)) if verbose else layers
    l2toks = {}
    prompt_len = prompt_tokens.shape[1]-1
    
    for i in lrange:
        toks = prompt_tokens.repeat(intervene_vec.shape[1], 1)
        start_len = toks.shape[1]
        probas_tok = []
        for idx_tok in range(n_new_tokens):
            T = toks.shape[1]
            token_index = intervene_tok-idx_tok if intervene_tok < 0 else intervene_tok

            with nnmodel.trace(toks, validate=False, scan=False):
                if len(intervene_vec.shape)>2:
                    nnmodel.model.layers[i].output[0][:, prompt_len:, :] += intervene_vec[i, :, :].repeat(toks.shape[1]-prompt_len,1,1).permute(1,0,2)
                else:
                    nnmodel.model.layers[i].output[0][:, prompt_len:, :] += intervene_vec[:, :].repeat(toks.shape[1]-prompt_len,1,1).permute(2,0,1)
                logits = nnmodel.lm_head.output[:, -1, :].save()
            probas_tok.append(logits.value.softmax(dim=-1).detach().cpu())
            pred_tok = t.argmax(logits.value, dim=-1, keepdim=True)
            toks = t.cat([toks, pred_tok.to(toks.device)], dim=-1)
            l2toks[i] = toks.detach().cpu()[:, start_len:]
        probas.append(t.stack(probas_tok))
    probas = t.stack(probas)
    probas = probas[:, :, 0]
    
    if None is not None:
        return [tokenizer.decode(t) for t in list(l2toks.values())[0]], probas
    return {k: [tokenizer.decode(t) for t in v] for k, v in l2toks.items()}, probas


def contrastive_act_gen_hooked(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    intervene_vec: t.Tensor,
    prompt: str,
    n_new_tokens: int = 10,
    layer: Optional[Union[int, List[int]]] = None,
    intervene_tok: int = -1,
    verbose: bool = False,
) -> (Dict[int, List[str]], t.Tensor):
    """
    Runs a contrastive generation by adding intervene_vec at the chosen residual stream(s)
    in HookedTransformer. Returns completions and token probabilities.

    Args:
        model: A HookedTransformer instance (TransformerLens).
        tokenizer: A HuggingFace tokenizer.
        intervene_vec: A tensor with shape:
                       - either (n_layers, batch_size, d_model) if multi-layer
                       - or (batch_size, d_model) if single-layer
                       This tensor is added to the residual stream at the chosen layer(s)
                       for the newly generated tokens.
        prompt: The text prompt to start generation from.
        n_new_tokens: Number of tokens to autoregressively generate.
        layer: Which layer(s) to intervene on. If None, will patch all layers.
        intervene_tok: The token index on which to apply the vector. If negative, counts backward from the end.
        verbose: If True, uses trange for progress display.

    Returns:
        A tuple:
          - A dictionary mapping layer_idx -> list of decoded completions (strings),
          - A tensor of shape (num_layers_being_patched, n_new_tokens, vocab_size) containing
            the probability distribution for each newly generated token.
    """

    device = model.cfg.device  # e.g. 'cuda' or 'cpu'
    model.to(device)
    model.eval()

    # Tokenize prompt
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_tokens.shape[1]

    # Decide which layers to patch
    if layer is None:
        layers_to_patch = list(range(model.cfg.n_layers))
    elif isinstance(layer, int):
        layers_to_patch = [layer]
    else:
        layers_to_patch = layer

    # We'll store, for each layer, a list of completions
    layer_to_completions = {}
    # We'll also store probabilities for each step (layer, step, vocab_size)
    layer_probs = []

    # For convenience: if intervene_vec has shape (n_layers, batch_size, d_model), 
    # we assume one "batch_size" dimension that might correspond to different runs.
    # If intervene_vec has shape (batch_size, d_model), we'll just broadcast it.

    # A small helper to broadcast intervene_vec if needed
    def get_intervene_slice(
        intervene_vec: t.Tensor,
        layer_idx: int,
        batch_idx: int,
        seq_len_diff: int
    ) -> t.Tensor:
        """
        Returns the slice of intervene_vec to be added to the residual at layer_idx.
        We'll broadcast across the newly generated tokens (seq_len_diff).
        """
        # If shape is (n_layers, batch_size, d_model):
        if intervene_vec.dim() == 3:
            # intervene_vec[i, b, d_model]
            return intervene_vec[layer_idx, batch_idx].unsqueeze(0).repeat(seq_len_diff, 1)
        # If shape is (batch_size, d_model):
        elif intervene_vec.dim() == 2:
            # intervene_vec[b, d_model]
            return intervene_vec[batch_idx].unsqueeze(0).repeat(seq_len_diff, 1)
        else:
            raise ValueError("intervene_vec must have shape (n_layers, B, d_model) or (B, d_model).")

    # We'll define a single-forward-step function that:
    # 1. Registers a hooking function that modifies the residual at the chosen layer.
    # 2. Runs the model forward to get next-token logits.
    # 3. Removes the hook (so we can re-register it for each new token if we want).
    def generate_next_token(
        current_tokens: t.Tensor, 
        layer_idx: int,
        batch_idx: int,
        offset: int
    ) -> t.Tensor:
        """
        Autoregressively generates the next token (argmax).
        offset is how many tokens we've already generated so far (0 to n_new_tokens-1).
        """
        seq_len_diff = current_tokens.shape[1] - prompt_len

        def patch_resid_post_hook(resid: t.Tensor, hook):
            # resid shape: [batch_size, seq_len, d_model]
            # We only want to patch the newly generated tokens, i.e. from prompt_len onward.
            patch_slice = get_intervene_slice(
                intervene_vec, 
                layer_idx, 
                batch_idx,
                seq_len_diff=seq_len_diff
            )
            # Add to the newly generated positions
            resid[:, prompt_len:, :] += patch_slice
            return resid

        # Forward pass with hooking
        with model.hooks(fwd_hooks=[
            (f"blocks.{layer_idx}.hook_resid_post", patch_resid_post_hook)
        ]):
            logits = model(current_tokens)  # [batch_size, seq_len, vocab_size]
        
        # Extract the logits for the last token in each batch row
        next_logits = logits[:, -1, :]  # shape: [batch_size, vocab_size]
        return next_logits

    # We iterate over each layer, do a batch generation, gather completions & probabilities
    iteration_range = trange if verbose else range
    for i in iteration_range(len(layers_to_patch)):
        layer_idx = layers_to_patch[i]
        # For demonstration, let's assume we only do a "single" batch at a time 
        # because your code repeats intervene_vec.shape[1] times. 
        # Here, we show the logic for a single batch dimension. 
        # If you have multiple batch dims in intervene_vec, adapt accordingly.

        # We store for each batch dimension, the final strings. If multiple, you'd do it in a loop.
        completions_for_layer = []
        
        # We'll do exactly one pass if intervene_vec.shape[0] is for multiple runs. 
        # Feel free to adapt for your scenario.
        batch_size = intervene_vec.shape[1] if intervene_vec.dim() == 3 else intervene_vec.shape[0]
        # We'll collect probabilities for each newly generated token
        # shape: (n_new_tokens, vocab_size) per batch item, but let's store them in a python list first.
        all_probs = []

        for batch_idx in range(batch_size):
            # Start fresh from the prompt
            toks = prompt_tokens.clone()
            
            # Generate token-by-token
            for step in range(n_new_tokens):
                # get next token logits
                next_logits = generate_next_token(toks, layer_idx, batch_idx, step)
                # store the softmax distribution
                dist = next_logits.softmax(dim=-1)  # [batch_size, vocab_size]

                # Suppose we only do 1 batch item at a time in this example:
                dist_0 = dist[0].detach().cpu()  # shape: [vocab_size]
                all_probs.append(dist_0)

                # Argmax next token
                next_token = dist_0.argmax(dim=-1, keepdim=True)  # [1]
                # Append to existing tokens
                next_token = next_token.unsqueeze(0).to(device)  # shape [1, 1]
                toks = t.cat([toks, next_token], dim=-1)

            # Decode final sequence (new tokens only or entire prompt). 
            # The example code decodes the newly generated tokens, so:
            new_toks = toks[0, prompt_len:]  # shape [n_new_tokens]
            completion_str = tokenizer.decode(new_toks)
            completions_for_layer.append(completion_str)

        # Convert list of distributions into a single tensor: [n_new_tokens, vocab_size]
        all_probs_tensor = t.stack(all_probs, dim=0)  # [n_new_tokens * batch_size, vocab_size]
        # If you have multiple batch items, you may want to reshape to [batch_size, n_new_tokens, vocab_size].
        # For simplicity, let's store them as is. We'll just keep the last one (if you only want the first item).
        layer_probs.append(all_probs_tensor)

        # Save completions
        layer_to_completions[layer_idx] = completions_for_layer

    # Finally, stack the probabilities along dimension 0 (one entry per layer):
    # shape: [num_layers, n_new_tokens * batch_size, vocab_size]
    probs = t.stack(layer_probs, dim=0)

    return layer_to_completions, probs