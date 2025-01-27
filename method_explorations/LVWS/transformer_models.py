
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout=0.0):
        super(MultiHeadTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections for multi-head Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection after multi-head concat
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def multi_head_attention(self, x, mask=None):
        """
        x: (batch_size, seq_len, embed_dim)
        mask: (batch_size, seq_len, seq_len) or None
        Returns: (batch_size, seq_len, embed_dim)
        """
        B, S, E = x.shape

        # Project to Q, K, V (each [B, S, E])
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Reshape to heads: [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention per head
        # Q @ K^T -> [B, num_heads, S, S]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, num_heads, S, head_dim]

        # Re-combine heads -> [B, S, embed_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(out)  # final linear proj

    def forward(self, x, mask=None):
        # Multi-head self-attention
        attn_out = self.multi_head_attention(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadTransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerScheduler(nn.Module):
    def __init__(self, robot_input_dimensions, task_input_dimension, embed_dim, ff_dim, num_heads, num_layers, dropout = 0.0):
        
        super(TransformerScheduler, self).__init__()
        self.robot_embedding = nn.Linear(robot_input_dimensions, embed_dim)
        self.task_embedding = nn.Linear(task_input_dimension, embed_dim)

        self.robot_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.task_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, num_heads, num_layers, dropout)

        # MLP to map [robot_embedding, task_embedding] -> single reward value
        self.reward_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)  # outputs scalar per (robot, task) pair
        )

    def forward(self, robot_features, task_features):
        """
        robot_features: shape (batch_size, n_robots, robot_input_dimensions)
        task_features:  shape (batch_size, n_tasks,  task_input_dimensions)

        returns: reward_matrix of shape (batch_size, n_robots, n_tasks)
        """
        B, N, _ = robot_features.shape  # e.g., B, n_robots, robot_input_dimensions
        _, M, _ = task_features.shape   # e.g., B, n_tasks,  task_input_dimensions

        # 1) Embed each robot/task
        # Result: (B, N, embed_dim) and (B, M, embed_dim)
        robot_emb = self.robot_embedding(robot_features)
        task_emb  = self.task_embedding(task_features)

        # 2) Pass through robot and task TransformerEncoders
        # Still shape: (B, N, embed_dim) and (B, M, embed_dim)
        robot_out = self.robot_transformer_encoder(robot_emb)  # shape: (B, N, embed_dim)
        task_out  = self.task_transformer_encoder(task_emb)    # shape: (B, M, embed_dim)

        # 3) Build pairwise combinations for robot-task
        # We'll expand both to shape (B, N, M, embed_dim) and then concat along last dim
        # so that each "pair" is [robot_embedding, task_embedding].
        # This yields a shape (B, N, M, 2*embed_dim).
        expanded_robot = robot_out.unsqueeze(2).expand(B, N, M, robot_out.shape[-1])
        expanded_task  = task_out.unsqueeze(1).expand(B, N, M, task_out.shape[-1])
        pairwise_input = torch.cat([expanded_robot, expanded_task], dim=-1)  # (B, N, M, 2*embed_dim)

        # 4) Run the MLP to get a scalar reward for each (robot, task) pair
        # shape after MLP: (B, N, M, 1), then squeeze -> (B, N, M)
        reward_scores = self.reward_mlp(pairwise_input)        # (B, N, M, 1)
        reward_scores = reward_scores.squeeze(-1)              # (B, N, M)

        return reward_scores