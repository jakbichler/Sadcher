import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

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

        # Linear transformation: W_e * h
        Wh = self.W(x)  # shape: (B, N, out_dim)

        # Compute pairwise combinations: [W_e h_i || W_e h_j]
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.out_dim)  # (B, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.out_dim)  # (B, N, N, out_dim)
        e_ij = self.a(torch.cat([Wh_i, Wh_j], dim=-1))  # (B, N, N, 1)

        # Mask non-adjacent nodes
        mask = (adj == 0).unsqueeze(-1)
        e_ij = e_ij.masked_fill(mask, float('-inf'))

        # Softmax over neighbor dimension (dim=2)
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
        
        # If we want the final output dimension to be `out_dim`, we typically
        # let each head produce out_dim // num_heads, then we concat them.
        assert out_dim % num_heads == 0, \
            "out_dim must be divisible by num_heads for concatenation."
        self.head_dim = out_dim // num_heads
        
        # Create each single-head GAT
        self.heads = nn.ModuleList([
            GraphAttentionHead(in_dim, self.head_dim, negative_slope=negative_slope)
            for _ in range(num_heads)
        ])

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
        # Pass through stacked multi-head layers
        for layer in self.layers:
            x = layer(x, adj)
        return x


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
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj   = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.LeakyReLU(negative_slope=-0.01),
            nn.Linear(ff_dim, embed_dim)
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

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
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
        self.layers = nn.ModuleList([
            MultiHeadTransformerBlock(embed_dim, ff_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

# -----------------------------
# TransformerScheduler
# -----------------------------

class SchedulerNetwork(nn.Module):
    def __init__(self, robot_input_dimensions, task_input_dimension, embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, n_gatn_heads, n_gatn_layers,  dropout=0.0):
        """
        robot_input_dimensions: Expected dimensions for robot features (first two must be (x,y))
        task_input_dimension: Expected dimensions for task features (first two must be (x,y))
        """
        super().__init__()

        self.robot_embedding = nn.Linear(robot_input_dimensions, embed_dim)
        self.task_embedding = nn.Linear(task_input_dimension, embed_dim)
        self.robot_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)
        self.task_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)

        self.robot_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, dropout)
        self.task_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, dropout)

        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Outputs a single scalar per (robot, task) pair.
        )

        # Rewards MLP for normal tasks 
        self.reward_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)  # outputs scalar per (robot, task) pair
        )

        # Rewards MLP for idle tasks
        self.idle_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1) # outputs scalar per robot
        )


    def forward(self, robot_features, task_features, task_adjacencies=None):
        """
        robot_features: Tensor of shape (B, N, robot_input_dimensions) where first 2 dims are (x,y)
        task_features:  Tensor of shape (B, M, task_input_dimension) where first 2 dims are (x,y)
        """
        B, N, _ = robot_features.shape
        _, M, _ = task_features.shape

        robot_emb = self.robot_embedding(robot_features)  # (B, N, embed_dim)
        task_emb  = self.task_embedding(task_features)       # (B, M, embed_dim)

        robot_gatn_output = self.robot_GATN(robot_emb, adj=None)  # (B, N, embed_dim)
        task_gatn_output  = self.task_GATN(task_emb, adj=task_adjacencies)    # (B, M, embed_dim)

        robot_out = self.robot_transformer_encoder(robot_gatn_output)  # (B, N, embed_dim)
        task_out  = self.task_transformer_encoder(task_gatn_output)    # (B, M, embed_dim)

        # 3) Build pairwise feature tensor.
        expanded_robot_gatn = robot_gatn_output.unsqueeze(2).expand(B, N, M, robot_gatn_output.shape[-1]) # (B, N, M, embed_dim)
        expanded_task_gatn  = task_gatn_output.unsqueeze(1).expand(B, N, M, task_gatn_output.shape[-1])  # (B, N, M, embed_dim)

        expanded_robot_out = robot_out.unsqueeze(2).expand(B, N, M, robot_out.shape[-1]) # (B, N, M, embed_dim)
        expanded_task_out  = task_out.unsqueeze(1).expand(B, N, M, task_out.shape[-1]) # (B, N, M, embed_dim)

        # 4) Compute pairwise relative distances from raw positions.
        #  the first two dimensions of the raw features are (x,y).
        robot_positions = robot_features[:, :, :2]  # (B, N, 2)
        task_positions  = task_features[:, :, :2]     # (B, M, 2)
        robot_pos_exp = robot_positions.unsqueeze(2).expand(B, N, M, 2)
        task_pos_exp  = task_positions.unsqueeze(1).expand(B, N, M, 2)
        rel_distance = torch.norm(robot_pos_exp - task_pos_exp, dim=-1, keepdim=True)
        rel_distance = rel_distance / torch.max(rel_distance)  # (B, N, M, 1)
        processed_distance = self.distance_mlp(rel_distance)  # (B, N, M, 1)

        # 5) Concatenate all features for the final reward MLP.
        final_input = torch.cat([expanded_robot_gatn, expanded_task_gatn, expanded_robot_out, expanded_task_out, processed_distance], dim=-1) # (B, N, M, 4*embed_dim + 1)

        task_rewards = self.reward_mlp(final_input).squeeze(-1)  # (B, N, M)
        idle_rewards_per_task = self.idle_mlp(final_input).squeeze(-1)  # (B, N, M)
        idle_rewards = idle_rewards_per_task.sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Concatenate the idle reward with task rewards, so final shape is (B, N, M+1)
        final_reward = torch.cat([task_rewards, idle_rewards], dim=-1)
        return final_reward
    