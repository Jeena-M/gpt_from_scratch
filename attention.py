import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Register causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores
        att = q @ k.transpose(-2, -1) / (C ** 0.5)

        # Apply causal mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float('inf'))

        # Normalize
        att = F.softmax(att, dim = 1)

        # Weighted sum of values
        out = att @ v   # (B, T, C)

        return out
    
if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 4, 8) # batch=2, tokens=4, embedding=8
    attn = CausalSelfAttention(embed_dim=8)

    y = attn(x)
    print(y.shape)

