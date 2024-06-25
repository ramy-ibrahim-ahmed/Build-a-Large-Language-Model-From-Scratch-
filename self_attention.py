"""
1. get queries by inputs @ w_query
2. get keys by inputs @ w_key
2. get values by inputs @ w_value
4. attn_scores by queries @ keys.T
5. normalization via softmax with attn_scores
    which normalized by 0.5 emb_dim and it is called (attn_weights)
6. get context_vec by attn_weights @ values

context matrix contain numbers corresponding to tokens
each number pos is contributed by each token
so each token pos in context M has been computed from all remain tokens.
"""

import torch
import torch.nn as nn

# Example input
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)


# Initializations
x_2 = inputs[1]  # second input element
d_in = inputs.shape[1]  # the input embedding size, d=3
d_out = 2  # the output embedding size, d=2
torch.manual_seed(123)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


# Get query key value vectors for token 2
query_2 = x_2 @ w_query  # 1x3 @ 3x2
key_2 = x_2 @ w_key
value_2 = x_2 @ w_value
# print(query_2)


# queries keys values
queries = inputs @ w_query  # 6x3 @ 3x2
keys = inputs @ w_key
values = inputs @ w_value


# attn scores for token 2 corresponding to token 2
key_2 = keys[1]
attn_score_22 = queries[1].dot(key_2)  #
# print(attn_score_22)  # 2 @ 2


# attn scores for all tokens corresponding to token 2.
# All attention scores for given query
attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)


# Normalization with softmax.
# on softmax normalize with 0.5 emb_dim for more stable probas
# giving a chance for small to be appear
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)


# Context vector
context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)  # 1x6 @ 6x2


# Self attention (scaled-dot product attention)
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)
        # attn scores
        attn_scores = queries @ keys.T
        # attn weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # context vectors
        return attn_weights @ values


torch.manual_seed(789)
self_attn = SelfAttention(d_in, d_out)
# print(self_attn(inputs))