"""
steps:
    1. get logits
    2. get propas by softmax
    3. get propas with actual targets
    4. log them
    5. mean log
    6. negative likelyhood to minimize

corss entropy from pytorch do all these steps.

preplexity is confusion next tken metrics.
it should equale valid tokens.
"""

import torch
from gpt2_archeticture import GPTModel, tokenizer
from text_generation import token_ids_to_text

# Congif with lower attn dimentions for ease of training
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}


# model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # Disable dropout during inference


# input target
inputs = torch.tensor(
    # ["every effort moves", " I really like"]
    [[16833, 3626, 6100], [40, 1107, 588]],
)
targets = torch.tensor(
    # [" effort moves you", " really like chocolate"]
    [[3626, 6100, 345], [1107, 588, 11311]],
)


# Get predictions propapilitys
with torch.no_grad():
    logits = model(inputs)

# propas for each token in vocab for its next token prediction
propas = torch.softmax(logits, dim=-1)
# (batch_size, num_tokens, vocab_size)
# print(propas.shape)


# Expected tokens
token_ids = torch.argmax(propas, dim=-1, keepdim=True)
# print(token_ids)


# Decode to text
# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


# Probabilities corresponding to the target token IDs
# first sentence, first 3 tokens, their corresponding propas for target idx
target_propas_1 = propas[0, [0, 1, 2], targets[0]]
# print(target_propas_1)
target_propas_2 = propas[1, [0, 1, 2], targets[1]]
# print(target_propas_2)


# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_propas_1, target_propas_2)))
# print(log_probas)


# Average log probabilities
avg_log_propas = torch.mean(log_probas)
# print(avg_log_propas)


# Negative likelyhood for minimization convention
neg_avg_log_propas = avg_log_propas * -1
# print(neg_avg_log_propas)


# (batch_size, num_tokens, vocab_size)
# print(logits.shape)
# (batch_size, num_tokens)
# print(targets.shape)
# flatten over batch for cross entropy
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten(0)
# print(logits_flat.shape)
# print(targets_flat.shape)


# Cross entropy
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)


# Perplexity
perplexity = torch.exp(loss)
# print(perplexity)
# our model was confused about 48725 token