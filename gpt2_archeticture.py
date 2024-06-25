"""
The token embedding layer
    projects the 50,257-dimensional one-hot encoded input
    tokens to a 768-dimensional embedding representation.

The output layer
    projects 768-dimensional embeddings back into a 50,257-dimensional representation
    so that we can convert these back into words.
"""

import torch
import torch.nn as nn
import tiktoken
from transformer_block import TransformerBlock
from layer_norm import LayerNorm

# prepare data and try dummy
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)


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


# GPT-2 version small archeticture
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeds = self.token_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        # Shape [batch_size, num_tokens, emb_size]
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


# try architecture
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)


# total parameters
total_params = sum(param.numel() for param in model.parameters())
# print(f"Total number of parameters: {total_params:,}")


# embedding weights & out head weights
# print("Token embedding layer shape:", model.token_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)


# total params as GPT-2 small paper share emb & out head weights
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")


# memory required for the model
# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4
# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")