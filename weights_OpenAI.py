import torch
import numpy as np
import urllib.request
from gpt_download import download_and_load_gpt2
from gpt2_archeticture import GPTModel, GPT_CONFIG_124M, tokenizer
from text_generation import text_to_token_ids, token_ids_to_text
from decodeing_strategies import generate

# Download py script for weights downloading
# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch05/"
#     "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split("/")[-1]
# urllib.request.urlretrieve(url, filename)


# Download from OpenAI
settings, params = download_and_load_gpt2(
    model_size="124M",  # small
    models_dir=r"D:\ramy\Tutorial\LLMS from scratch --book\OpenAI",
)


# print("Settings:", settings)
# print("Parameter dictionary keys:", params.keys())


# print(params["wte"])
# print("Token embedding weight tensor dimensions:", params["wte"].shape)


# Configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})


# Instance
gpt2_small = GPTModel(NEW_CONFIG)
gpt2_small.eval()


# Ensure competability in weights tensors shapes
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# Load weights carefully
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].attn.w_query.weight = assign(
            gpt.transformer_blocks[b].attn.w_query.weight, q_w.T
        )
        gpt.transformer_blocks[b].attn.w_key.weight = assign(
            gpt.transformer_blocks[b].attn.w_key.weight, k_w.T
        )
        gpt.transformer_blocks[b].attn.w_value.weight = assign(
            gpt.transformer_blocks[b].attn.w_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].attn.w_query.bias = assign(
            gpt.transformer_blocks[b].attn.w_query.bias, q_b
        )
        gpt.transformer_blocks[b].attn.w_key.bias = assign(gpt.transformer_blocks[b].attn.w_key.bias, k_b)
        gpt.transformer_blocks[b].attn.w_value.bias = assign(
            gpt.transformer_blocks[b].attn.w_value.bias, v_b
        )

        gpt.transformer_blocks[b].attn.out_proj.weight = assign(
            gpt.transformer_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].attn.out_proj.bias = assign(
            gpt.transformer_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.transformer_blocks[b].ffn.layers[0].weight = assign(
            gpt.transformer_blocks[b].ffn.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[b].ffn.layers[0].bias = assign(
            gpt.transformer_blocks[b].ffn.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.transformer_blocks[b].ffn.layers[2].weight = assign(
            gpt.transformer_blocks[b].ffn.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].ffn.layers[2].bias = assign(
            gpt.transformer_blocks[b].ffn.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


device = "cpu"
load_weights_into_gpt(gpt2_small, params)
gpt2_small.to(device)


# try itt_
torch.manual_seed(123)
token_ids = generate(
    model=gpt2_small,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
torch.save(gpt2_small.state_dict(), "GPTV2_small.pth")