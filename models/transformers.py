import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Transformer Building Blocks
# -----------------------------


class MultiHeadTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout=0.0):
        super(MultiHeadTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.LeakyReLU(negative_slope=-0.01),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def multi_head_attention(self, x, mask=None):
        B, S, E = x.shape
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(out)

    def forward(self, x, mask=None):
        attn_out = self.multi_head_attention(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                MultiHeadTransformerBlock(embed_dim, ff_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
