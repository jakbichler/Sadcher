
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.0):
        """
        Initializes a Transformer Block with a single-head attention mechanism.

        Args:
            embed_dim (int): Size of input embeddings.
            ff_dim (int): Size of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super(SingleHeadTransformerBlock, self).__init__()
        self.embed_dim = embed_dim

        # Linear layers for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask=None):
        """
        Compute the scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, embed_dim).
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # (batch_size, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        return torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_dim)

    def forward(self, x, mask=None):
        """
        Forward pass of the Transformer Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Attention
        Q = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.key_proj(x)    # (batch_size, seq_len, embed_dim)
        V = self.value_proj(x)  # (batch_size, seq_len, embed_dim)

        attention_output = self.attention(Q, K, V, mask)  # (batch_size, seq_len, embed_dim)
        attention_output = self.out_proj(attention_output)

        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class TransformerScheduler(nn.Module):
    def __init__(self, robot_input_dimensions, task_input_dimension, embed_dim, ff_dim, num_layers, dropout = 0.0):
        
        super(TransformerScheduler, self).__init__()
        self.robot_embedding = nn.Linear(robot_input_dimensions, embed_dim)
        self.task_embedding = nn.Linear(task_input_dimension, embed_dim)

        self.transformer_blocks = nn.ModuleList([
            SingleHeadTransformerBlock(embed_dim, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, robot_features, task_features):
        robot_embeddings = self.robot_embedding(robot_features)
        task_embeddings = self.task_embedding(task_features)

        x = torch.cat([robot_embeddings, task_embeddings], dim=1)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        n_robots = robot_features.shape[1] 

        features_per_robot = x[:, :n_robots]
        features_per_task = x[:, n_robots:]
        reward_matrix = torch.matmul(features_per_robot, features_per_task.transpose(1,2))
        
        return reward_matrix 