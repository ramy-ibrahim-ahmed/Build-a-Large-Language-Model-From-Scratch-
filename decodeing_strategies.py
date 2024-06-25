import torch
import matplotlib.pyplot as plt
from training import GPT_CONFIG_124M, model, tokenizer
from text_generation import text_to_token_ids, token_ids_to_text

# set model to cpu because of inference
# turn off random components (dropout)
model.to("cpu")
model.eval()


# vocab
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}


# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)


# Greedy decoding
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])


# Propabilistic sampling
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
# print(inverse_vocab[next_token_id])


# not every time forward will be selected
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


# print_sampled_tokens(probas)


# Temperature scaling
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


# Original, higher confidence, and lower confidence
temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]


# plot
# x = torch.arange(len(vocab))
# bar_width = 0.15
# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(
#         x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
#     )
# ax.set_ylabel("Probability")
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
# plt.tight_layout()
# plt.savefig("temperature-plot.pdf")
# plt.grid(True, lw=0.5, linestyle="--")
# plt.show()


# print_sampled_tokens(scaled_probas[1])
# print_sampled_tokens(scaled_probas[2])


# Top K sampling
# top k ids
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# print("Top logits:", top_logits)
# print("Top positions:", top_pos)


# -inf masking
new_logits = torch.where(
    # top_logits[-1] most small one
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits,
)
# print(new_logits)


# softmax propas
topk_propas = torch.softmax(new_logits, dim=0)
# print(topk_propas)


# text generator
def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    # Get logits
    # only focus on last time step
    for _ in range(max_new_tokens):
        # valid tokens to consider last "context size"
        # (batch_size, num_tokens)
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():  # dont compute gradients
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # keep only top k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

            # Temperature
            if temperature > 0.0:
                logits = logits / temperature
                # (batch_size, context_len)
                probs = torch.softmax(logits, dim=-1)
                # (batch_size, 1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # (batch_size, 1)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop when next token is eos(end of sentence)
            if idx_next == eos_id:
                break

            # Append next token to all context
            # (batch_size, num_tokens+1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


# try it
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))