"""
Non efficent approtch
    1. normalize scores.
    2. mask matrix with context len with 0s up diagonal.
    3. multiply mask 0s by (normalied) weights.
    4. normalize again with mean division.

Efficent approtch
    1. mask via -inf up diagonal.
    2. normalization with softmax results 0s on -infs.

Causal attention make the next token prediction based on
paset tokens not up coming one that mimice the real senario
during inference.

Dropout make it more harder which make training more greate adjustment.
"""

import torch
import torch.nn as nn
from self_attention import self_attn

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


# attention weights computation
queries = self_attn.w_query(inputs)
keys = self_attn.w_key(inputs)
attn_scores = queries @ keys.T
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
# print(attn_weights)


# zeros up diagonal mask
context_length = attn_scores.shape[0]
mask_zeros = torch.tril(torch.ones(context_length, context_length))
# print(mask_zeros)


# Get masked causal attention weights
masked_zeros = attn_weights * mask_zeros
# print(masked_zeros)


# Additional normalization for ensure row sum to 1 (propas)
row_sums = masked_zeros.sum(dim=1, keepdim=True)
masked_zeros_norm = masked_zeros / row_sums
# print(masked_zeros_norm)


# Mask -inf once before softmax
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)


# Apply softmax normalization
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
# print(attn_weights)


# Dropout masking
# drop on fraq randomly and multibly by 1/fraq
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # dropout rate of 50%
example = torch.ones(6, 6)  # create a matrix of ones
# print(dropout(example))


# 2 inputs with 6 tokens each, and each token has embedding dimension 3
batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout_rate, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1),
        )

    def forward(self, x):
        batches, num_tokens, d_in = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)
        # transpose over tokens
        attn_scores = queries @ keys.transpose(1, 2)
        # masking
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # normalization
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # dropout
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


# Try it
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)
