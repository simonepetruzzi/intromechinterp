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
#from eindex import eindex
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

device = t.device("cuda" if t.cuda.is_available() else "cpu")

#import part2_intro_to_mech_interp.tests as tests
#from plotly_utils import hist, imshow, plot_comp_scores, plot_logit_attribution, plot_loss_difference

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

#import the model from the TransformerLens through HookedTransformer
gpt2_small = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Load model and return its parameters
# model_description_text = """## Loading Models

# HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

# For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""


# loss = gpt2_small(model_description_text, return_type="loss")
# print("Model loss:", loss)
# logits = gpt2_small(model_description_text, return_type="logits")
# print("Model logits:", logits)


#Cache the activations 
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))

# 2 ways to extract the attention pattern for layer 0
# attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
# attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

# t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)

# Visualize attention heads
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
# Attention visualization
visualization = cv.attention.attention_heads(
    tokens=gpt2_str_tokens,
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(12)],
)

# Export visualization to an HTML file
with open("attention_heads_visualization.html", "w") as f:
    f.write(str(visualization))
