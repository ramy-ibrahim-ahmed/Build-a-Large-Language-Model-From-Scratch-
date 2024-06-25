"""
Skip connections
    preserve the magnitude of the previous layer's activations,
    which are then fed into the next layer.
    This helps prevent vanishing gradients
    and allows for a smoother flow of information through the network.

GELU (Gaussian Error Linear Unit) 
    is an activation function that is less prone to saturation 
    compared to other functions like ReLU. 
    This helps maintain gradients during training.

Skip connections also allow for 
the direct transfer of information from earlier layers to later layers, 
without being significantly transformed.
This preserves important information 
that might be lost in deeper layers.

Layer normalization
    helps to stabilize the activations within a layer 
    by normalizing their values. 
    This prevents them from becoming too large or too small
"""

import torch
import torch.nn as nn
from feed_forward_network import GELU


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


# nn with skip connections
class ResidualDeepNetwork(nn.Module):
    def __init__(self, layers_neurons, use_shortcut):
        super().__init__()
        self.use_sortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layers_neurons[0], layers_neurons[1]), GELU()),
                nn.Sequential(nn.Linear(layers_neurons[1], layers_neurons[2]), GELU()),
                nn.Sequential(nn.Linear(layers_neurons[2], layers_neurons[3]), GELU()),
                nn.Sequential(nn.Linear(layers_neurons[3], layers_neurons[4]), GELU()),
                nn.Sequential(nn.Linear(layers_neurons[4], layers_neurons[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            if self.use_sortcut and out.shape == x.shape:
                x = x + out
            else:
                x = out
        return x


# def to compute gradients mean throw backward in layers
def print_gradients(net, x):
    out = net(x)
    target = torch.tensor([[0.0]])

    loss = nn.MSELoss()
    loss = loss(out, target)

    # function for computing gradients for layers
    # without math manual code needed to compute
    loss.backward()

    for name, param in net.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# try net without skip connections
layers_neurons = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
net_without_shortcuts = ResidualDeepNetwork(
    layers_neurons,
    use_shortcut=False,
)
# print_gradients(net_without_shortcuts, sample_input)


# net with skip connections
torch.manual_seed(123)
net_without_shortcuts = ResidualDeepNetwork(
    layers_neurons,
    use_shortcut=True,
)
# print_gradients(net_without_shortcuts, sample_input)