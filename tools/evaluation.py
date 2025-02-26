import torch as t
from torch import Tensor
import torch.nn.functional as F
import pandas as pd
import re
from collections import Counter

def calculate_perplexity(logits: Tensor, target_tokens: Tensor) -> float:
    # Get the shape of the logits tensor
    _, _, vocab_size = logits.shape

    # Reshape logits and targets
    logits_flat = logits.view(-1, vocab_size)   # Shape: (8, 5)
    targets_flat = target_tokens.view(-1)       # Shape: (8)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')

    # Compute perplexity
    perplexity = t.exp(loss)

    return perplexity.item()


def get_answer_type_final(row, check_for = "index", format="{0}"):
    if check_for=="index":
        ans_en = str(int(row["ans_west_idx"]))
        ans_tr = str(int(row["ans_local_idx"]))
    else:
        ans_en = format.format(row["ans_west"]).lower().strip()
        ans_tr = format.format(row["ans_local"]).lower().strip()
    
    row["ans_type"] = "none"
    out = str(row["output"]).lower().strip()
    if ans_en in out and ans_tr in out:
        en_index = out.index(ans_en)
        tr_index = out.index(ans_tr)
        row["ans_type"] = "west" if en_index < tr_index else "local"
    elif ans_en in out:
        row["ans_type"] = "west"
    elif ans_tr in out:
        row["ans_type"] = "local"

    return row
