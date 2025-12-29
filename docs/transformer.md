# Transformer Block (GPT-Style, Pre-LayerNorm)

This project builds a GPT model from scratch. The fundamental unit of GPT is the **Transformer block**, which combines:
1) **Causal self-attention** (token-to-token communication)
2) **Feed-forward network (MLP)** (per-token computation)
along with **LayerNorm** and **residual connections** for stable deep training.

---

## 1. High-Level Structure

A GPT transformer block is:

1. `x = x + Attention(LayerNorm(x))`
2. `x = x + MLP(LayerNorm(x))`

This is the **Pre-LayerNorm** design used in GPT-style models.

---

## 2. Why Residual Connections?

Residual (skip) connections (`x + f(x)`) are essential because they:

- Stabilize optimization in deep networks
- Improve gradient flow (reducing vanishing gradients)
- Let each block learn a *correction* to the current representation rather than rewriting it entirely

Conceptually, a block learns:  
> “What should I add to the current representation to make it better?”

---

## 3. Why LayerNorm?

LayerNorm normalizes activations across the embedding dimension for each token. It:

- Keeps activation scales stable across layers
- Makes training less sensitive to learning rates and initialization
- Prevents attention/MLP outputs from becoming numerically unstable

In **Pre-LN** transformers, normalization happens *before* attention/MLP, which tends to make deep stacks easier to train.

---

## 4. Attention vs. MLP: Communication vs. Computation

A transformer block has two distinct roles:

### Causal Self-Attention = Communication
Attention lets each token selectively pull information from previous tokens in the sequence.  
This is how the model becomes context-aware (e.g., resolving pronouns, tracking topics).

### MLP = Computation
The MLP processes each token independently (no mixing across tokens).  
It is responsible for nonlinear feature transformation and representation learning.

A helpful mental model:
- **Attention = routing**
- **MLP = compute**

Stacking blocks alternates these two operations, enabling expressive sequence modeling.

---

## 5. Why the MLP Expands to 4× the Embedding Size?

The standard GPT-style MLP uses:

- `C → 4C → C`

The expansion creates a larger intermediate space to apply nonlinear transformations (via GELU) before projecting back. This increases representational capacity without increasing sequence length.

---

## 6. Implementation Notes

In `transformer.py`, the block is implemented as:

- `ln1` + `CausalSelfAttention` + residual
- `ln2` + `MLP` (Linear → GELU → Linear) + residual

The output shape matches the input shape `(B, T, C)`, enabling stacking multiple blocks to form a full GPT model.
