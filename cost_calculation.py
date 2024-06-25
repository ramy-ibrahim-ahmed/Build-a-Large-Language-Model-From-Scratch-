"""
data loder and split for training validation
calculate batch loss cross entropy
calculate cost of num batches
"""

import torch
from dataloader_sliding_window import tokenizer, create_dataloader_v1
from cross_entropy import GPT_CONFIG_124M, model

file_path = "./the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)


# Split dataset
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


# Dataloaders
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
)


# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)


# train_tokens = 0
# for input_batch, target_batch in train_loader:
#     train_tokens += input_batch.numel()
# val_tokens = 0
# for input_batch, target_batch in val_loader:
#     val_tokens += input_batch.numel()
# print("Training tokens:", train_tokens)
# print("Validation tokens:", val_tokens)
# print("All tokens:", train_tokens + val_tokens)


# Loss for a given batch
def loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )


def cost(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# Try it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = cost(train_loader, model, device)
    val_loss = cost(val_loader, model, device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)