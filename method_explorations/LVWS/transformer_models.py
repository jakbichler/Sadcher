import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Graph Attention Building Blocks
# -----------------------------
class GraphAttentionHead(nn.Module):
    """
    A single-head GAT layer (head) that updates each node embedding
    by attending to its neighbors in the adjacency matrix.
    """
    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Learnable linear transform for node features:
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention projection a: from (out_dim + out_dim) -> scalar
        # We'll do [W h_i || W h_j] -> a -> e_ij
        self.a = nn.Linear(2*out_dim, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x, adj):
        """
        x:   (B, N, in_dim) -- node embeddings for B graphs, each with N nodes
        adj: (B, N, N)      -- adjacency (0/1) for each pair of nodes
        Returns: updated node embeddings of shape (B, N, out_dim)
        """
        B, N, _ = x.shape
        
        # 1) Linear transformation of node features
        Wh = self.W(x)  # shape (B, N, out_dim)
        # 2) Prepare pairwise combination for attention: [Wh_i || Wh_j]
        #    We'll tile Wh so that each node can attend to every other node:
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.out_dim)  # shape (B, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.out_dim)  # shape (B, N, N, out_dim)
        # Concatenate along features
        e_ij = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, N, N, 2*out_dim)
        
        # 3) Compute attention logits
        e_ij = self.leaky_relu(self.a(e_ij))  # (B, N, N, 1)
        
        # 4) Mask out non-edges with -inf (so they vanish in softmax)
        mask = (adj == 0).unsqueeze(-1)  # shape (B, N, N, 1)
        e_ij = e_ij.masked_fill(mask, float('-inf'))
        
        # 5) Softmax over neighbors j
        alpha_ij = F.softmax(e_ij, dim=2)  # (B, N, N, 1)
        
        # 6) Compute final updated node embeddings by summing over neighbors
        h_prime = alpha_ij * Wh_j  # (B, N, N, out_dim)
        h_prime = h_prime.sum(dim=2)  # sum over j -> (B, N, out_dim)
        
        return h_prime

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
# Unified Multi-Head Attention Block
# -----------------------------
class MultiHeadAttention(nn.Module):
    """
    A unified multi-head attention module that can be used for self-attention (Q=K=V)
    as well as cross-attention (with different Q, K, V).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        B, L, _ = query.size()
        _, S, _ = key.size()
        
        Q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, L, head_dim)
        K = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)    # (B, heads, S, head_dim)
        V = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, L, S)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out)

# -----------------------------
# Transformer Building Blocks using Unified Attention
# -----------------------------
class MultiHeadTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.LeakyReLU(negative_slope=-0.01),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, x, x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return self.norm2(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadTransformerBlock(embed_dim, ff_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

# -----------------------------
# Bidirectional Cross-Transformer Block using Unified Attention
# -----------------------------
class BidirectionalCrossTransformerBlock(nn.Module):
    """
    A bidirectional cross-attention block that updates both robot and task embeddings.
    For the robot branch, queries come from robot tokens and keys/values come from task tokens;
    for the task branch, queries come from task tokens and keys/values come from robot tokens.
    """
    def __init__(self, embed_dim, ff_dim, num_heads, dropout=0.0):
        super().__init__()
        # Cross attention modules for each branch
        self.robot_cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.task_cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm_robot = nn.LayerNorm(embed_dim)
        self.norm_task = nn.LayerNorm(embed_dim)

        self.ffn_robot = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ffn_task = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm_ffn_robot = nn.LayerNorm(embed_dim)
        self.norm_ffn_task = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, robot, task):
        # Robot branch: robot queries attend to task keys/values
        robot_attn = self.robot_cross_attention(robot, task, task)
        robot_attn = self.dropout(robot_attn)
        robot_updated = self.norm_robot(robot + robot_attn)
        ffn_robot = self.ffn_robot(robot_updated)
        ffn_robot = self.dropout(ffn_robot)
        robot_final = self.norm_ffn_robot(robot_updated + ffn_robot)

        # Task branch: task queries attend to robot keys/values
        task_attn = self.task_cross_attention(task, robot, robot)
        task_attn = self.dropout(task_attn)
        task_updated = self.norm_task(task + task_attn)
        ffn_task = self.ffn_task(task_updated)
        ffn_task = self.dropout(ffn_task)
        task_final = self.norm_ffn_task(task_updated + ffn_task)

        return robot_final, task_final

class CrossTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, num_layers=1, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalCrossTransformerBlock(embed_dim, ff_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, robot, task):
        for layer in self.layers:
            robot, task = layer(robot, task)
        return robot, task

# -----------------------------
# SchedulerNetwork with Unified Multi-Head Attention
# -----------------------------
class SchedulerNetwork(nn.Module):
    def __init__(self, robot_input_dimensions, task_input_dimension, embed_dim, ff_dim,
                 n_transformer_heads, n_transformer_layers, n_gatn_heads, n_gatn_layers,
                 n_cross_layers=1, dropout=0.0):
        """
        robot_input_dimensions: expected dims for robot features (first two are (x,y))
        task_input_dimension: expected dims for task features (first two are (x,y))
        """
        super().__init__()
        self.robot_embedding = nn.Linear(robot_input_dimensions, embed_dim)
        self.task_embedding = nn.Linear(task_input_dimension, embed_dim)

        self.robot_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)
        self.task_GATN  = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)

        self.robot_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, dropout)
        self.task_transformer_encoder  = TransformerEncoder(embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, dropout)

        # Cross encoder replaces original transformer outputs with cross-attended representations.
        self.cross_encoder = CrossTransformerEncoder(embed_dim, ff_dim, n_transformer_heads, num_layers=n_cross_layers, dropout=dropout)

        # MLP to process relative distance (first two dims in raw features are (x,y))
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Final reward MLP: concatenates GATN outputs and cross-attended outputs + processed distance.
        self.reward_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)
        )

    def forward(self, robot_features, task_features):
        """
        robot_features: (B, N, robot_input_dimensions), first 2 dims are (x,y)
        task_features:  (B, M, task_input_dimension), first 2 dims are (x,y)
        """
        B, N, _ = robot_features.shape
        _, M, _ = task_features.shape

        # Embedding
        robot_emb = self.robot_embedding(robot_features)  # (B, N, embed_dim)
        task_emb  = self.task_embedding(task_features)       # (B, M, embed_dim)

        # Intra-modal processing with GATN and transformer encoders
        robot_gatn_output = self.robot_GATN(robot_emb, adj=None)  # (B, N, embed_dim)
        task_gatn_output  = self.task_GATN(task_emb, adj=None)     # (B, M, embed_dim)

        robot_trans = self.robot_transformer_encoder(robot_gatn_output)  # (B, N, embed_dim)
        task_trans  = self.task_transformer_encoder(task_gatn_output)    # (B, M, embed_dim)

        # Cross attention: update representations via bidirectional cross encoder.
        robot_cross, task_cross = self.cross_encoder(robot_trans, task_trans)

        # Build pairwise feature tensor for reward computation.
        expanded_robot_gatn = robot_gatn_output.unsqueeze(2).expand(B, N, M, robot_gatn_output.size(-1))
        expanded_task_gatn  = task_gatn_output.unsqueeze(1).expand(B, N, M, task_gatn_output.size(-1))
        expanded_robot_cross = robot_cross.unsqueeze(2).expand(B, N, M, robot_cross.size(-1))
        expanded_task_cross  = task_cross.unsqueeze(1).expand(B, N, M, task_cross.size(-1))

        # Compute normalized pairwise relative distances from raw positions (first two dims: x,y)
        robot_positions = robot_features[:, :, :2]  # (B, N, 2)
        task_positions  = task_features[:, :, :2]     # (B, M, 2)
        robot_pos_exp = robot_positions.unsqueeze(2).expand(B, N, M, 2)
        task_pos_exp  = task_positions.unsqueeze(1).expand(B, N, M, 2)
        rel_distance = torch.norm(robot_pos_exp - task_pos_exp, dim=-1, keepdim=True)
        rel_distance = rel_distance / torch.max(rel_distance)
        processed_distance = self.distance_mlp(rel_distance)  # (B, N, M, 1)

        # Concatenate all features and compute reward scores.
        final_input = torch.cat([
            expanded_robot_gatn,
            expanded_task_gatn,
            expanded_robot_cross,
            expanded_task_cross,
            processed_distance
        ], dim=-1)  # (B, N, M, 4*embed_dim+1)
        reward_scores = self.reward_mlp(final_input).squeeze(-1)  # (B, N, M)
        return reward_scores
    