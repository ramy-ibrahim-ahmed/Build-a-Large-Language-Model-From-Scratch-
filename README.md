# Building GPT-2 from Scratch for Text Generation

This repository demonstrates how to build a GPT-2 model from scratch, utilizing a sliding window technique for efficient text generation. Below, we break down the components and architecture involved in the process.

## Transformer Block

The transformer block is the core building unit of the GPT-2 architecture. It consists of a multi-head self-attention mechanism followed by position-wise feed-forward networks. The block also includes layer normalization and residual connections to ensure stable and efficient training.

![Transformer Block](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/13.webp?1)

**Key Components:**
- **Multi-head Self-Attention:** Allows the model to focus on different parts of the input sequence simultaneously.
- **Feed-Forward Networks:** Apply transformations to each position independently.
- **Layer Normalization:** Normalizes the output to improve training stability.
- **Residual Connections:** Help in gradient flow and mitigate the vanishing gradient problem.

## GPT-2 Architecture

GPT-2 (Generative Pre-trained Transformer 2) is an autoregressive model designed to generate coherent and contextually relevant text. The small variant of GPT-2 retains the core architecture but with fewer layers and parameters, making it suitable for demonstration and learning purposes.

![GPT-2 Small Architecture](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/gpt-sizes.webp?timestamp=123)

**Architecture Highlights:**
- **Byte Pair Encoding (BPE) Tokenizer:** Efficiently tokenizes text by iteratively merging the most frequent pairs of bytes in a corpus.
- **Token Embedding:** Converts input tokens into dense vectors of fixed size, capturing semantic meaning.
- **Positional Encoding:** Injects information about the position of tokens in the sequence, crucial for understanding the order.
- **Stacked Transformer Blocks:** Multiple layers of transformer blocks to increase model capacity.
- **Sliding Window Technique:** Efficiently handles long sequences by processing chunks of text while maintaining context.
