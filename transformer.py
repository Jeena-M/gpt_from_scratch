import torch
import torch.nn as nn
from attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        # attention block
        x = x + self.attn(self.ln1(x))

        # feedforward block
        x = x + self.mlp(self.ln2(x))

        return x
    
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    B, T, C = 2, 4, 8
    x = torch.randn(B, T, C)

    block = TransformerBlock(embed_dim=C)
    y = block(x)

    print("x:", x.shape)
    print("y:", y.shape)
