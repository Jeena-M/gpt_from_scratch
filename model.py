import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, max_seq_len=256, n_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(n_layers)])

        # Final normalization + language modeling head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token ids
        targets: (B, T) token ids, optional
        """
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        # (B, T, C)
        tok = self.token_emb(idx)

        # (T, C) broadcast to (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Logits for next-token prediction: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T)
            loss = F.cross_entropy(logits.view(B * T, self.vocab_size), targets.view(B * T))

        return logits, loss


if __name__ == "__main__":
    torch.manual_seed(0)

    # Tiny smoke test
    vocab_size = 65
    model = GPT(vocab_size=vocab_size, embed_dim=32, max_seq_len=16, n_layers=2)

    idx = torch.randint(0, vocab_size, (2, 8))  # (B=2, T=8)
    logits, loss = model(idx, targets=idx)
    print("Any NaNs in logits?", torch.isnan(logits).any().item())
    print("Any NaNs in embeddings?", torch.isnan(model.token_emb(idx)).any().item())

    print("logits:", logits.shape)  # (2, 8, vocab_size)
    print("loss:", loss.item())
