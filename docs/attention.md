# Causal Self-Attention

This document explains the theory, intuition, and implementation of **causal self-attention**, the core mechanism behind GPT-style large language models.

This module was implemented from scratch in `attention.py` as part of a bottom-up GPT implementation.

---

## 1. Why Attention Is Needed

Earlier sequence models (RNNs, LSTMs, GRUs) process tokens sequentially and compress all past information into a single hidden state.

This creates two major problems:

1. **Information bottleneck**  
   Distant tokens are harder to remember, even with gating mechanisms.

2. **Lack of parallelism**  
   Tokens must be processed one at a time, limiting scalability.

Attention solves both problems by allowing **each token to directly access all previous tokens** in a single operation.

---

## 2. What Is Self-Attention?

Self-attention allows each token in a sequence to decide:
> *Which other tokens in this same sequence are relevant to me right now?*

- **Self**: tokens attend to other tokens in the same input
- **Attention**: relevance-weighted information flow
- **Causal**: tokens cannot attend to future tokens

This makes the model **autoregressive**, which is essential for language generation.

---

## 3. Query, Key, Value (Intuition)

Each token is projected into three vectors:

| Vector | Role | Intuition |
|------|------|----------|
| Query (Q) | What I am looking for | “What kind of information do I need?” |
| Key (K) | What I contain | “What kind of information do I offer?” |
| Value (V) | What I give | “What information should be passed along?” |

Attention works by:
1. Comparing queries with keys
2. Turning similarities into weights
3. Using those weights to combine values

This allows relevance to be **learned dynamically**, not hard-coded.

---

## 4. Mathematical Formulation

Given an input tensor:

x ∈ ℝ^(B × T × C)

Where:
- `B` = batch size
- `T` = sequence length
- `C` = embedding dimension

We compute:

Q = xW_q
K = xW_k
V = xW_v


Attention scores:

scores = QKᵀ / √C


The scaling factor √C prevents large dot products from pushing softmax into saturation.

After applying a **causal mask** and softmax:

weights = softmax(scores)
output = weights · V


The output has the same shape as the input:

output ∈ ℝ^(B × T × C)


---

## 5. Causal Masking

Causal masking ensures that token *t* can only attend to tokens ≤ *t*.

Without this:
- The model would see future tokens during training
- Generation would break at inference time

In implementation, this is done using a **lower-triangular mask**, where future positions are set to `-∞` before softmax.

This enforces strict left-to-right information flow.

---

## 6. Implementation Walkthrough

Key implementation details in `attention.py`:

- **Linear projections**  
  Separate `nn.Linear` layers are used for Q, K, and V to allow the model to learn different representations.

- **No bias terms**  
  Biases are unnecessary here and are commonly omitted in transformer implementations.

- **Registering the mask as a buffer**  
  The causal mask is stored as a non-trainable tensor that moves automatically to the correct device (CPU/GPU).

- **Shape preservation**  
  Attention mixes information across tokens but preserves tensor shape, enabling stacking of layers.

---

## 7. Why Attention Is Powerful

Causal self-attention is:

- **Context-aware**: relevance is computed dynamically per token
- **Parallelizable**: all tokens processed at once
- **Scalable**: enables deep transformer stacks
- **General**: same mechanism used in NLP, vision, and multimodal models

This single mechanism replaced most previous sequence modeling approaches.

---

## 8. Key Takeaways

- Attention allows tokens to selectively use context
- Q/K/V projections enable flexible relevance matching
- Causal masking is essential for autoregressive models
- Shape preservation enables deep architectures like GPT

Understanding this module means understanding the core of modern LLMs.
