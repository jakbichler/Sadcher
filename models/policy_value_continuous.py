import sys

import numpy as np
from icecream import ic

sys.path.append("..")

import torch
import torch.nn as nn
from skrl.envs.wrappers.torch.gymnasium_envs import unflatten_tensorized_space
from skrl.models.torch import DeterministicMixin, Model, MultivariateGaussianMixin

from models.graph_attention import GATEncoder
from models.scheduler_network import SchedulerNetwork
from models.transformers import TransformerEncoder


class ContinuousSchedulerPolicy(MultivariateGaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        policy_config,
        pretrained=False,
        use_idle=True,
        predict_stddev=True,
        debug=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        log_stddev_init=-0.5,
    ):
        MultivariateGaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        Model.__init__(self, observation_space, action_space, device)
        self.device = device
        self.use_idle = use_idle
        self.debug = debug

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
            self.load_pretrained_weights(checkpoint_path)
            self.freeze_encoder_layers()

        if predict_stddev:
            action_dim = np.prod(action_space.shape)
            self.log_stddev = nn.Parameter(
                torch.ones(action_dim, device=self.device) * log_stddev_init, requires_grad=True
            )

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

        batch_size = robot_features.shape[0]
        n_robots = robot_features.shape[1]
        n_tasks = task_features.shape[1]
        n_actions = n_tasks + 1 if self.use_idle else n_tasks

        mu = self.scheduler_net(
            robot_features, task_features, task_adjacency
        )  # shape: (batch, n_robot , n_actions)

        mu = mu.view(batch_size, n_robots * n_actions)

        return mu, self.log_stddev, {}

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_state_dict = checkpoint.get(
            "state_dict", checkpoint
        )  # In case the checkpoint is wrapped

        current_state_dict = self.scheduler_net.state_dict()

        filtered_checkpoint_state_dict = {
            k: v
            for k, v in checkpoint_state_dict.items()
            if k in current_state_dict and v.size() == current_state_dict[k].size()
        }

        skipped_layers = [
            k
            for k, v in checkpoint_state_dict.items()
            if k not in current_state_dict or v.size() != current_state_dict[k].size()
        ]

        self.scheduler_net.load_state_dict(filtered_checkpoint_state_dict, strict=False)

        if skipped_layers:
            print("Skipped layers due to shape mismatch or missing in the new model:")
            for layer in skipped_layers:
                print(f"  - {layer}")
        else:
            print("No layers were skipped.")

        print(f"Loaded {len(filtered_checkpoint_state_dict)} matching layers from checkpoint.")
        print(f"Skipped {len(skipped_layers)} layers.")

    def freeze_encoder_layers(self):
        frozen_count = 0
        trainable_count = 0

        for param in self.scheduler_net.robot_GATN.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
        for param in self.scheduler_net.task_GATN.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()

        for param in self.scheduler_net.robot_transformer_encoder.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
        for param in self.scheduler_net.task_transformer_encoder.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()

        for param in self.scheduler_net.robot_embedding.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
        for param in self.scheduler_net.task_embedding.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()

        for param in self.scheduler_net.parameters():
            if param.requires_grad:
                trainable_count += param.numel()

        total_params = frozen_count + trainable_count
        print(f"Total parameters in the scheduler network: {total_params}")
        print(f"Frozen parameters (elements): {frozen_count}")
        print(f"Trainable parameters (elements): {trainable_count}")

        print("All encoder layers in the scheduler network have been frozen.")

    # ====================================
    # Custom Value Network
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


class ZeroCritic(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

    def compute(self, states, role):
        batch = states["states"].shape[0]
        return torch.zeros(batch, 1, device=self.device), {}
