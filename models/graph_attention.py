import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Graph Attention Building Blocks
# -----------------------------
class GraphAttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x, adj):
        B, N, _ = x.shape
        device = x.device

        # Add self-loops
        self_loops = torch.eye(N, device=device).expand(B, N, N)
        adj = (adj + self_loops).clamp(max=1)

        Wh = self.W(x)  # (B, N, out_dim)

        # Compute pairwise combinations: [W_e h_i || W_e h_j]
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.out_dim)  # (B, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.out_dim)  # (B, N, N, out_dim)
        e_ij = self.a(torch.cat([Wh_i, Wh_j], dim=-1))  # (B, N, N, 1)

        # Mask non-adjacent nodes
        mask = (adj == 0).unsqueeze(-1)
        e_ij = e_ij.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e_ij, dim=2)  # (B, N, N, 1)

        # Self-loop contribution: α_{i,i} * W_e h_i
        alpha_self = torch.diagonal(alpha.squeeze(-1), dim1=1, dim2=2).unsqueeze(-1)
        self_contrib = alpha_self * Wh  # (B, N, out_dim)

        # Neighbor contribution: LeakyReLU(∑_{j≠i} α_{i,j} * W_e h_j)
        mask_diag = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1)
        alpha_neighbors = alpha.masked_fill(mask_diag, 0)
        neighbor_sum = torch.sum(alpha_neighbors * Wh_j, dim=2)  # (B, N, out_dim)
        neighbor_contrib = self.leaky_relu(neighbor_sum)  # (B, N, out_dim)
        return self_contrib + neighbor_contrib


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    A multi-head GAT layer that runs multiple single-head GATs in parallel
    and concatenates their outputs.
    """

    def __init__(self, in_dim, out_dim, num_heads=4, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads for concatenation."
        self.head_dim = out_dim // num_heads

        self.heads = nn.ModuleList(
            [
                GraphAttentionHead(in_dim, self.head_dim, negative_slope=negative_slope)
                for _ in range(num_heads)
            ]
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        """
        x:   (B, N, in_dim)
        adj: (B, N, N)
        returns: (B, N, out_dim)
           where out_dim = num_heads * head_dim
        """
        # Each head returns (B, N, head_dim)
        head_outputs = [head(x, adj) for head in self.heads]
        # Concatenate along the feature dimension
        # shape => (B, N, num_heads * head_dim) = (B, N, out_dim)
        out = torch.cat(head_outputs, dim=-1)  # (B, N, num_heads * head_dim)

        return self.norm(x + out)


class GATEncoder(nn.Module):
    """
    A small GAT network that can have multiple multi-head layers.
    Each multi-head layer -> (B, N, embed_dim).
    """

    def __init__(self, embed_dim, num_heads=4, num_layers=2, negative_slope=0.2):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                MultiHeadGraphAttentionLayer(embed_dim, embed_dim, num_heads, negative_slope)
            )

    def forward(self, x, adj):
        if adj is None:
            adj = torch.ones(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for layer in self.layers:
            x = layer(x, adj)
        return x
