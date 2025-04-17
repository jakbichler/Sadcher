import sys

import numpy as np

sys.path.append("..")

import torch
import torch.nn as nn
from skrl.envs.wrappers.torch.gymnasium_envs import unflatten_tensorized_space
from skrl.models.torch import DeterministicMixin, Model, MultiCategoricalMixin

from models.graph_attention import GATEncoder
from models.scheduler_network import SchedulerNetwork
from models.transformers import TransformerEncoder


class SchedulerPolicy(MultiCategoricalMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        policy_config,
        pretrained=False,
        use_idle=True,
        use_positional_encoding=False,
    ):
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob=False, reduction="sum")  #
        Model.__init__(self, observation_space, action_space, device)
        self.device = device
        self.use_idle = use_idle
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            robot_input_dimensions = policy_config["robot_input_dimensions"] + 1
        else:
            robot_input_dimensions = policy_config["robot_input_dimensions"]

        task_input_dimension = policy_config["task_input_dimension"]
        embed_dim = policy_config["embed_dim"]
        ff_dim = policy_config["ff_dim"]
        n_transformer_heads = policy_config["n_transformer_heads"]
        n_transformer_layers = policy_config["n_transformer_layers"]
        n_gatn_heads = policy_config["n_gatn_heads"]
        n_gatn_layers = policy_config["n_gatn_layers"]

        self.scheduler_net = SchedulerNetwork(
            robot_input_dimensions=robot_input_dimensions,
            task_input_dimension=task_input_dimension,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            n_transformer_heads=n_transformer_heads,
            n_transformer_layers=n_transformer_layers,
            n_gatn_heads=n_gatn_heads,
            n_gatn_layers=n_gatn_layers,
            use_idle=use_idle,
        ).to(self.device)

        if pretrained:
            checkpoint_path = "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
            self.scheduler_net.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    def compute(self, states, taken_actions=None, timesteps=None, eval=False, **kwargs):
        """
        input: a dictionary with keys "robot_features" and "task_features"
          - robot_features: [batch, n_robots, 7] (last element is availability)
          - task_features: [batch, n_tasks, 9] (index 6 is the ready flag)

        """
        if not eval:
            states = unflatten_tensorized_space(self.observation_space, states["states"])

        robot_features = states["robot_features"].to(self.device)  # shape: (batch, n_robots, 7)
        task_features = states["task_features"].to(self.device)  # shape: (batch, n_tasks, 9)
        task_adjacency = states["task_adjacency"].to(
            self.device
        )  # shape: (batch, n_tasks, n_tasks)

        if self.use_positional_encoding:  # Append noramlized robot id
            batch_size, n_robots, input_dim = robot_features.shape
            robot_ids = (
                torch.arange(n_robots).unsqueeze(0).expand(batch_size, n_robots).to(self.device)
            )  # Shape: (B, N)
            normalized_robot_ids = robot_ids / (n_robots - 1)  # Normalize to the range [0, 1]
            robot_features = torch.cat(
                [robot_features, normalized_robot_ids.unsqueeze(-1)], dim=-1
            )  # Shape: (B, N, robot_input_dimensions + 1)
        # For robots: last element is availability (1: available, 0: not)
        robot_availability = robot_features[:, :, -1]  # shape: (batch, n_robots)
        # For tasks: assume index -3 is the ready flag last three are (ready, assigned, incomplete)
        task_incomplete = task_features[:, :, -1]  # shape: (batch, n_tasks)
        task_ready = task_features[:, :, -3]  # shape: (batch, n_tasks)
        task_ready_incomplete = torch.logical_and(task_ready, task_incomplete)

        batch_size = robot_features.shape[0]
        n_robots = robot_features.shape[1]
        n_tasks = task_features.shape[1]

        if self.use_idle:
            # Our action space is defined as n_tasks + 1 (normal + IDLE)
            n_actions = n_tasks + 1
        else:
            n_actions = n_tasks

        # Build a mask of shape (batch, n_robots, n_actions) that indicates valid actions.
        mask = torch.ones((batch_size, n_robots, n_actions), device=self.device)
        availability_mask = robot_availability.unsqueeze(-1).to(self.device)  # (batch, n_robots, 1)
        ready_incomplete_mask = task_ready_incomplete.unsqueeze(1).to(
            self.device
        )  # (batch, 1, n_tasks)
        task_mask = availability_mask * ready_incomplete_mask  # (batch, n_robots, n_tasks)

        if self.use_idle:
            # Insert the task_mask into the valid action indices (until -1 for IDLE, always ready).
            mask[:, :, :-1] = task_mask
        else:
            mask = task_mask

        reward_matrix = self.scheduler_net(
            robot_features, task_features, task_adjacency
        )  # shape: (batch, n_robots, n_tasks + 1)
        logits = reward_matrix.masked_fill(mask == 0, -1e9).float()
        probas = torch.softmax(logits, dim=-1)  # shape: (batch, n_robots, n_tasks + 1)
        net_output = probas.view(batch_size, n_robots * n_actions)

        # Assuming probas is already calculated as the softmax probabilities
        top_k_values, top_k_indices = probas.topk(
            5, dim=-1
        )  # Get top 5 values and indices along the last dimension

        ## Print the top 5 probabilities for each robot rounded to 2 decimals
        # for i in range(batch_size):  # Loop over each batch
        # for j in range(n_robots):  # Loop over each robot in the batch
        # top_5_probs = (
        # top_k_values[i, j].cpu().numpy()
        # )  # Get top 5 probabilities for robot j
        # top_5_tasks = top_k_indices[i, j].cpu().numpy()  # Get corresponding task indices
        # print(f"  Robot {j}:")
        # for k in range(5):
        # print(f"    Task {top_5_tasks[k] + 1}: {top_5_probs[k]:.2f}")
        # print()

        return net_output, {}

    # ====================================
    # Custom Value Network (No Masking Needed)
    # ====================================


class SchedulerValue(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        value_config,
        clip_actions=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model.__init__(self, observation_space, action_space, self.device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        robot_input_dim = value_config["robot_input_dim"]
        task_input_dim = value_config["task_input_dim"]
        embed_dim = value_config["embed_dim"]
        ff_dim = value_config["ff_dim"]
        n_transformer_heads = value_config["n_transformer_heads"]
        n_transformer_layers = value_config["n_transformer_layers"]
        n_gatn_heads = value_config["n_gatn_heads"]
        n_gatn_layers = value_config["n_gatn_layers"]
        self.robot_embedding = nn.Linear(robot_input_dim, embed_dim)
        self.task_embedding = nn.Linear(task_input_dim, embed_dim)
        self.robot_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)
        self.task_GATN = GATEncoder(embed_dim, n_gatn_heads, n_gatn_layers)

        self.robot_transformer_encoder = TransformerEncoder(
            embed_dim,
            ff_dim,
            n_transformer_heads,
            n_transformer_layers,
        )
        self.task_transformer_encoder = TransformerEncoder(
            embed_dim,
            ff_dim,
            n_transformer_heads,
            n_transformer_layers,
        )

        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Outputs a single scalar per (robot, task) pair.
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1),  # outputs scalar per robot
        )

    def compute(self, states, role):
        states = unflatten_tensorized_space(self.observation_space, states["states"])
        # Flatten the observation.
        robot_features = states["robot_features"].to(self.device)
        task_features = states["task_features"].to(self.device)
        task_adjacency = states["task_adjacency"].to(self.device)

        B, N, _ = robot_features.shape
        _, M, _ = task_features.shape

        robot_emb = self.robot_embedding(robot_features)  # (B, N, embed_dim)
        task_emb = self.task_embedding(task_features)  # (B, M, embed_dim)

        robot_gatn_output = self.robot_GATN(robot_emb, adj=None)  # (B, N, embed_dim)
        task_gatn_output = self.task_GATN(task_emb, adj=task_adjacency)  # (B, M, embed_dim)

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

        values = self.value_mlp(final_input).squeeze(-1)  # (B, N, M)
        values = values.mean(dim=(-2, -1), keepdim=True).squeeze(-1)  # shape: (B,1)

        return values, {}
