"""
for each epoch
    for each step
        reset gradients
        compute loss
        compute gradients
        update weights

evaluate model on all train & val data after each 5 steps
calculateing the mean of 5 batches.

generate sample with start context each epoch.

typically we use one epoch over much larger dataset.
"""

import torch
import matplotlib.pyplot as plt
from gpt2_archeticture import GPTModel, tokenizer
from cost_calculation import loss_batch, cost, train_loader, val_loader, device
from text_generation import text_to_token_ids, token_ids_to_text, generate_text_sample

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


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # losses and token seen trakers
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # main training loop
    for epoch in range(num_epochs):
        # set model to train mood
        model.train()

        for input_batch, target_batch in train_loader:
            # reset loss graidients to 0 from previous
            optimizer.zero_grad()
            loss = loss_batch(input_batch, target_batch, model, device)
            # calculate loss gradients
            loss.backward()
            # update optimizer weights with loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # evaluation every eval_freq time
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # generate sample after each epoch
        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
        )
    return train_losses, val_losses, track_tokens_seen


# evaluate model
def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter,
):
    model.eval()
    with torch.no_grad():
        train_loss = cost(
            train_loader,
            model,
            device,
            num_batches=eval_iter,
        )
        val_loss = cost(
            val_loader,
            model,
            device,
            num_batches=eval_iter,
        )
    model.train()
    return train_loss, val_loss


# generate sample during training (each epoch)
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_sample(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        decoded = token_ids_to_text(token_ids, tokenizer)
        print(decoded.replace("\n", " "))
    model.train()


# runn_
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     device,
#     num_epochs=num_epochs,
#     eval_freq=5,
#     eval_iter=5,
#     start_context="Every effort moves you",
#     tokenizer=tokenizer,
# )


# plot
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.grid(True, lw=0.5, linestyle="--")
    # plt.savefig("loss-plot.pdf")
    # plt.show()


# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)