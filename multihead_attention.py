import torch
import torch.nn as nn
from causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # each head take the input and make calculations
        # process done sequentially
        # head take x finish
        # head take x finish
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_len, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        # stack all heads as result context matrix
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        # Each head dim
        self.d_out = d_out
        self.num_heads = num_heads
        # if 4 dim needed in out and we have 2 haeds
        # so each head will produce 2 dims.
        self.head_dim = d_out // num_heads

        # query, key and value weights.
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # masking
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # (batches, tokens, embeddings)
        b, num_tokens, d_in = x.shape

        # (batches, tokens, d_out-(head_dim))
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        # split matriciesby add dim num_heads before head_dim
        # (batches, tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose (batches, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Keys transpose
        # (batches, num_heads, head_dim, num_tokens)
        # (tokens X head_dim) @ (head_dim X tokens)
        # every token query multiply by every token key
        # (1x2x3x4)@(1x2x4x3) -> (1x2x3x3)
        # attn_scores (batches, num_heads, num_tokens, head_dim)
        attn_scores = queries @ keys.transpose(2, 3)

        # Original mask truncated to the number of tokens and converted to boolean
        # Use the mask to fill attention scores
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalization & Dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batches, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Optional projection
        return self.out_proj(context_vec)


# explaination
a = torch.tensor(
    [
        [
            [
                [0.2745, 0.6584, 0.2775, 0.8573],
                [0.8993, 0.0390, 0.9268, 0.7388],
                [0.7179, 0.7058, 0.9156, 0.4340],
            ],
            [
                [0.0772, 0.3565, 0.1479, 0.5331],
                [0.4066, 0.2318, 0.4545, 0.9737],
                [0.4606, 0.5159, 0.4220, 0.5786],
            ],
        ]
    ]
)
# (batches, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
# print((a @ a.transpose(2, 3)))
# print((a @ a.transpose(2, 3)).shape)


first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
# print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
# print("\nSecond head:\n", second_res)