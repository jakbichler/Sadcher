import torch
import torch.nn as nn

from .graph_attention import GATEncoder
from .transformers import TransformerEncoder


class SchedulerNetwork(nn.Module):
    def __init__(
        self,
        robot_input_dimensions,
        task_input_dimension,
        embed_dim,
        ff_dim,
        n_transformer_heads,
        n_transformer_layers,
        n_gatn_heads,
        n_gatn_layers,
        dropout=0.0,
        use_idle=True,
    ):
        """
        robot_input_dimensions: Expected dimensions for robot features (first two must be (x,y))
        task_input_dimension: Expected dimensions for task features (first two must be (x,y))
        """
        super().__init__()
        self.use_idle = use_idle

        self.robot_embedding = nn.Linear(robot_input_dimensions, embed_dim)
        self.task_embedding = nn.Linear(task_input_dimension, embed_dim)
        self.robot_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)
        self.task_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)

        self.robot_transformer_encoder = TransformerEncoder(
            embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, dropout
        )
        self.task_transformer_encoder = TransformerEncoder(
            embed_dim, ff_dim, n_transformer_heads, n_transformer_layers, dropout
        )

        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Outputs a single scalar per (robot, task) pair.
        )

        # Rewards MLP for normal tasks
        self.reward_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1),  # outputs scalar per (robot, task) pair
        )

        if self.use_idle:
            # Rewards MLP for idle tasks
            self.idle_mlp = nn.Sequential(
                nn.Linear(4 * embed_dim + 1, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, 1),  # outputs scalar per robot
            )

    def forward(self, robot_features, task_features, task_adjacencies=None):
        """
        robot_features: Tensor of shape (B, N, robot_input_dimensions) where first 2 dims are (x,y)
        task_features:  Tensor of shape (B, M, task_input_dimension) where first 2 dims are (x,y)
        """
        B, N, _ = robot_features.shape
        _, M, _ = task_features.shape

        robot_emb = self.robot_embedding(robot_features)  # (B, N, embed_dim)
        task_emb = self.task_embedding(task_features)  # (B, M, embed_dim)

        robot_gatn_output = self.robot_GATN(robot_emb, adj=None)  # (B, N, embed_dim)
        task_gatn_output = self.task_GATN(task_emb, adj=task_adjacencies)  # (B, M, embed_dim)

        robot_out = self.robot_transformer_encoder(robot_gatn_output)  # (B, N, embed_dim)
        task_out = self.task_transformer_encoder(task_gatn_output)  # (B, M, embed_dim)

        # 3) Build pairwise feature tensor.
        expanded_robot_gatn = robot_gatn_output.unsqueeze(2).expand(
            B, N, M, robot_gatn_output.shape[-1]
        )  # (B, N, M, embed_dim)
        expanded_task_gatn = task_gatn_output.unsqueeze(1).expand(
            B, N, M, task_gatn_output.shape[-1]
        )  # (B, N, M, embed_dim)

        expanded_robot_out = robot_out.unsqueeze(2).expand(
            B, N, M, robot_out.shape[-1]
        )  # (B, N, M, embed_dim)
        expanded_task_out = task_out.unsqueeze(1).expand(
            B, N, M, task_out.shape[-1]
        )  # (B, N, M, embed_dim)

        # 4) Compute pairwise relative distances from raw positions.
        #  the first two dimensions of the raw features are (x,y).
        robot_positions = robot_features[:, :, :2]  # (B, N, 2)
        task_positions = task_features[:, :, :2]  # (B, M, 2)
        robot_pos_exp = robot_positions.unsqueeze(2).expand(B, N, M, 2)
        task_pos_exp = task_positions.unsqueeze(1).expand(B, N, M, 2)
        rel_distance = torch.norm(robot_pos_exp - task_pos_exp, dim=-1, keepdim=True)
        rel_distance = rel_distance / torch.max(rel_distance)  # (B, N, M, 1)
        processed_distance = self.distance_mlp(rel_distance)  # (B, N, M, 1)

        final_input = torch.cat(
            [
                expanded_robot_gatn,
                expanded_task_gatn,
                expanded_robot_out,
                expanded_task_out,
                processed_distance,
            ],
            dim=-1,
        )  # (B, N, M, 4*embed_dim + 1)

        task_rewards = self.reward_mlp(final_input).squeeze(-1)  # (B, N, M)

        if self.use_idle:
            idle_rewards_per_task = self.idle_mlp(final_input).squeeze(-1)  # (B, N, M)
            idle_rewards = idle_rewards_per_task.sum(dim=-1, keepdim=True)  # (B, N, 1)

            # Concatenate the idle reward with task rewards, so final shape is (B, N, M+1)
            return torch.cat([task_rewards, idle_rewards], dim=-1)

        else:
            return task_rewards
