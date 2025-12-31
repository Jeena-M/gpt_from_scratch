# GPT Model (`model.py`)

This module implements a minimal **decoder-only Transformer (GPT)** for causal language modeling.

---

## Overview

The model maps token IDs to next-token logits using:

1. Token embeddings  
2. Positional embeddings  
3. A stack of Transformer blocks  
4. A final normalization + linear projection  

It follows the standard GPT architecture.

---

## Architecture

tokens → embeddings → transformer blocks → layer norm → logits


Each transformer block performs:
- causal self-attention (context mixing)
- feed-forward computation (per-token transformation)
- residual connections and layer normalization

---

## Components

### Token Embeddings
Maps token IDs to dense vectors:
```python
nn.Embedding(vocab_size, embed_dim)

Positional Embeddings

Adds order information to tokens:
nn.Embedding(max_seq_len, embed_dim)

Transformer Stack

A sequence of identical blocks that refine token representations using attention + MLP.
self.blocks = nn.ModuleList([...])

Output Head
self.lm_head = nn.Linear(embed_dim, vocab_size)


Projects hidden states to logits over the vocabulary.

Forward Pass

Given input idx ∈ ℕ^(B×T):
1. Convert tokens → embeddings
2. Add positional embeddings
3. Pass through transformer blocks
4. Normalize
5. Project to vocabulary logits
6. Optionally compute cross-entropy loss

Training Objective

Uses causal language modeling:

The model learns to predict the next token at every position.

Loss:

CrossEntropy(logits, targets)