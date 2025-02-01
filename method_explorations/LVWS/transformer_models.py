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
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj   = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
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

class TransformerScheduler(nn.Module):
    def __init__(self, robot_input_dimensions, task_input_dimension, embed_dim, ff_dim, num_heads, num_layers, dropout=0.0):
        """
        robot_input_dimensions: Expected dimensions for robot features (first two must be (x,y))
        task_input_dimension: Expected dimensions for task features (first two must be (x,y))
        """
        super(TransformerScheduler, self).__init__()

        # Embedding layers for raw features.
        self.robot_embedding = nn.Linear(robot_input_dimensions, embed_dim)
        self.task_embedding = nn.Linear(task_input_dimension, embed_dim)

        # Transformer encoders for robot and task embeddings.
        self.robot_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, num_heads, num_layers, dropout)
        self.task_transformer_encoder = TransformerEncoder(embed_dim, ff_dim, num_heads, num_layers, dropout)

        # Dedicated sub-network to process the relative distance.
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Outputs a single scalar per (robot, task) pair.
        )

        # The final reward MLP now takes robot+task embeddings concatenated with the processed distance.
        self.reward_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)  # outputs scalar per (robot, task) pair
        )

    def forward(self, robot_features, task_features):
        """
        robot_features: Tensor of shape (B, N, robot_input_dimensions) where first 2 dims are (x,y)
        task_features:  Tensor of shape (B, M, task_input_dimension) where first 2 dims are (x,y)
        """
        B, N, _ = robot_features.shape
        _, M, _ = task_features.shape

        # 1) Embed robots and tasks.
        robot_emb = self.robot_embedding(robot_features)  # (B, N, embed_dim)
        task_emb  = self.task_embedding(task_features)       # (B, M, embed_dim)

        # 2) Process through transformer encoders.
        robot_out = self.robot_transformer_encoder(robot_emb)  # (B, N, embed_dim)
        task_out  = self.task_transformer_encoder(task_emb)    # (B, M, embed_dim)

        # 3) Build pairwise feature tensor.
        expanded_robot = robot_out.unsqueeze(2).expand(B, N, M, robot_out.shape[-1])
        expanded_task  = task_out.unsqueeze(1).expand(B, N, M, task_out.shape[-1])
        pairwise_features = torch.cat([expanded_robot, expanded_task], dim=-1)  # (B, N, M, 2*embed_dim)

        # 4) Compute pairwise relative distances from raw positions.
        #  the first two dimensions of the raw features are (x,y).
        robot_positions = robot_features[:, :, :2]  # (B, N, 2)
        task_positions  = task_features[:, :, :2]     # (B, M, 2)
        robot_pos_exp = robot_positions.unsqueeze(2).expand(B, N, M, 2)
        task_pos_exp  = task_positions.unsqueeze(1).expand(B, N, M, 2)
        # Euclidean distance along last dimension, output shape: (B, N, M, 1)
        rel_distance = torch.norm(robot_pos_exp - task_pos_exp, dim=-1, keepdim=True)
        rel_distance = rel_distance / torch.max(rel_distance)  # (B, N, M, 1)

        # 5) Process the distance through the dedicated MLP.
        processed_distance = self.distance_mlp(rel_distance)  # (B, N, M, 1)

        # 6) Concatenate processed distance with the pairwise embeddings.
        final_pairwise_input = torch.cat([pairwise_features, processed_distance], dim=-1)  # (B, N, M, 2*embed_dim+1)

        # 7) Compute reward scores.
        reward_scores = self.reward_mlp(final_pairwise_input).squeeze(-1)  # (B, N, M)

        return torch.clamp(reward_scores, min = 1e-6) ## In rare cases, model mayu predict negative values ---> DBGM will not work
