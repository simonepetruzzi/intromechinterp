import functools
import sys
from pathlib import Path
from typing import Callable
import html

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
import transformer_lens
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint
from huggingface_hub import hf_hub_download

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# This model has only attention heads and no MLP, no LayerNorm and no biases.
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# Visualize and inspect attention heads

# text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
# logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# def current_attn_detector(cache: ActivationCache) -> list[str]:
#     """
#     Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
#     """
#     attn_heads = []
#     for layer in range(model.cfg.n_layers):
#         for head in range(model.cfg.n_heads):
#             attention_pattern = cache["pattern", layer][head]
#             # take avg of diagonal elements
#             score = attention_pattern.diagonal().mean()
#             if score > 0.4:
#                 attn_heads.append(f"{layer}.{head}")
#     return attn_heads


# def prev_attn_detector(cache: ActivationCache) -> list[str]:
#     """
#     Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
#     """
#     attn_heads = []
#     for layer in range(model.cfg.n_layers):
#         for head in range(model.cfg.n_heads):
#             attention_pattern = cache["pattern", layer][head]
#             # take avg of sub-diagonal elements
#             score = attention_pattern.diagonal(-1).mean()
#             if score > 0.4:
#                 attn_heads.append(f"{layer}.{head}")
#     return attn_heads


# def first_attn_detector(cache: ActivationCache) -> list[str]:
#     """
#     Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
#     """
#     attn_heads = []
#     for layer in range(model.cfg.n_layers):
#         for head in range(model.cfg.n_heads):
#             attention_pattern = cache["pattern", layer][head]
#             # take avg of 0th elements
#             score = attention_pattern[:, 0].mean()
#             if score > 0.4:
#                 attn_heads.append(f"{layer}.{head}")
#     return attn_heads


# print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
# print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
# print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# Check wheter the model has induction circuits, namely it can predict well the token for the first hald of the sequence and less well the second half.

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens, logits, cache). This
    function should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache


# def get_log_probs(
#     logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
# ) -> Float[Tensor, "batch posn-1"]:
#     logprobs = logits.log_softmax(dim=-1)
#     batch_size, seq_len, _ = logits.size()
#     # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
#     correct_logprobs = logprobs[t.arange(batch_size).unsqueeze(1), t.arange(seq_len - 1).unsqueeze(0), tokens[:, 1:]]
#     return correct_logprobs


# seq_len = 50
# batch_size = 1
# (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
# rep_cache.remove_batch_dim()
# rep_str = model.to_str_tokens(rep_tokens)
# model.reset_hooks()
# log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

# print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
# print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")


# Detect induction-heads

def induction_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len + 1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()

print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))


