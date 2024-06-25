"""
- feed forward nn addes non linearity to the linear transformations
    which attn produce and discover more benifit features

- GPT architecture contains transformers blocks secutially
    this adds a type of horizontal processing during training
    which is benificial in fnn to extract features horizontally
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# GPT 2 configurations
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


# The GELU activation function is xΦ(x) --computationally expensive
# where Φ(x) the standard Gaussian cumulative distribution function
# GELU(x) ≈ 0.5 ⋅ x ⋅ (1 + tanh[√((2/π)) ⋅ (x + 0.044715 ⋅ x^3]) --for GPT
# smooth not like relu in zero corner isseus
# gives small negative values
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


# gelu, relu = GELU(), nn.ReLU()
# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function", fontsize=16)
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True, lw=0.5, linestyle="--")
# plt.tight_layout()
# plt.show()


# feed forward nn with expand dims for feed Gelu
# compose to original as the output of ffn
# that gives a wider representation for gelu to extract features
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# input shape: [batch_size, num_token, emb_size]
# 2 sentences, 3 words, 768 embeddings
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
# print(out.shape)