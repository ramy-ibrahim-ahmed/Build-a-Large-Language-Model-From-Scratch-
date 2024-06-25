"""
1. take context
2. feed to model to assigne next word predictions to each word
3. softmax predictions of last token which contains next word needs to be generated
4. argmax and add to context for next iter

model suppose to give each token the predictions of its next token
by consider all last supported context with attn
and enhanced attn predictions by ffn and the whole model components
"""

import torch
from gpt2_archeticture import tokenizer, model, GPT_CONFIG_124M, GPTModel


# function to generate text
def generate_text_sample(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        # print(idx.shape)

        # get predictions
        with torch.no_grad():
            logits = model(idx_cond)
            # print(logits.shape)

        # focus only on last token
        # logits (batch, n_tokens, vocab_size)
        # logits -1 gets last token with->(batch, vocab_size)
        logits = logits[:, -1, :]
        # print(logits.shape)

        # propabilities by softmax on dim of vocabs to be expected
        # (batch, vocab_size)
        propas = torch.softmax(logits, dim=-1)
        # print(propas.shape)

        # index of the heighest probability token to predict
        # (batch, 1)
        idx_next = torch.argmax(propas, dim=-1, keepdim=True)
        # print(idx_next.shape)

        # append sampled index to the running sequence
        # (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=-1)
        # print(idx.shape)
    return idx


# try our function
# encode the tokens of the context input
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
# print(encoded)


# tensor conversion for input for feed it to model
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print(encoded.shape)  # (1, 4)


# use model in iters
model.eval()  # disable dropout
out = generate_text_sample(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)
# print(out)
# print(len(out[0]))


# decode back to words
decoded = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded)


# congif with lower attn dimentions for ease of training
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


# converters
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # add batch dimention
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    # remove batch dimention
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# try converters
start_text = "Every effort moves you"
token_ids = generate_text_sample(
    model=model,
    idx=text_to_token_ids(start_text, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
# print(token_ids_to_text(token_ids, tokenizer))