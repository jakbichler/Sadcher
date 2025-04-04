import sys

sys.path.append("..")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.envs.wrappers.torch.gymnasium_envs import unflatten_tensorized_space
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model, MultiCategoricalMixin
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG

from models.scheduler_network import SchedulerNetwork


class SchedulerPolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob=False, reduction="sum")  #
        Model.__init__(self, observation_space, action_space, device)
        self.device = device
        self.scheduler_net = SchedulerNetwork(
            robot_input_dimensions=7,
            task_input_dimension=9,
            embed_dim=256,
            ff_dim=512,
            n_transformer_heads=4,
            n_transformer_layers=2,
            n_gatn_heads=8,
            n_gatn_layers=1,
        ).to(self.device)

        checkpoint_path = "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
        self.scheduler_net.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    def compute(self, states, taken_actions, timesteps=None, **kwargs):
        """
        input: a dictionary with keys "robot_features" and "task_features"
          - robot_features: [batch, n_robots, 7] (last element is availability)
          - task_features: [batch, n_tasks, 9] (index 6 is the ready flag)

        """
        states = unflatten_tensorized_space(self.observation_space, states["states"])
        robot_features = states["robot_features"].to(self.device)  # shape: (batch, n_robots, 7)
        task_features = states["task_features"].to(self.device)  # shape: (batch, n_tasks, 9)
        task_adjacency = states["task_adjacency"].to(
            self.device
        )  # shape: (batch, n_tasks, n_tasks)

        # print(f"ROBOT FEATURES : {robot_features}")
        # print(f"TASK FEATURES : {task_features}")
        # print(f"TASK ADJACENCY : {task_adjacency}")

        # For robots: last element is availability (1: available, 0: not)
        robot_availability = robot_features[:, :, -1]  # shape: (batch, n_robots)
        # For tasks: assume index -3 is the ready flag last three are (ready, assigned, incomplete)
        task_ready = task_features[:, :, -3]  # shape: (batch, n_tasks)

        batch_size = robot_features.shape[0]
        n_robots = robot_features.shape[1]
        n_tasks = task_features.shape[1]
        # Our action space is defined as n_tasks + 1 (normal + IDLE)
        n_actions = n_tasks + 1

        # Build a mask of shape (batch, n_robots, n_actions) that indicates valid actions.
        mask = torch.ones((batch_size, n_robots, n_actions), device=self.device)
        availability_mask = robot_availability.unsqueeze(-1).to(self.device)  # (batch, n_robots, 1)
        ready_mask = task_ready.unsqueeze(1).to(self.device)  # (batch, 1, n_tasks)
        task_mask = availability_mask * ready_mask  # (batch, n_robots, n_tasks)
        # Insert the task_mask into the valid action indices (until -1 for IDLE, always ready).
        mask[:, :, :-1] = task_mask

        # print(f"MASK SHAPE: {mask}")

        reward_matrix = self.scheduler_net(
            robot_features, task_features, task_adjacency
        )  # shape: (batch, n_robots, n_tasks + 1)
        # print(f"REWARD MATRIX SHAPE: {reward_matrix}")
        # Apply hard masking: set logits for invalid actions to a large negative value.
        logits = reward_matrix.masked_fill(mask == 0, -1e9)
        logits = torch.softmax(logits, dim=-1)  # shape: (batch, n_robots, n_tasks + 1)
        # Flatten the logits to shape (B, n_robots * n_actions) for the MultiCategoricalMixin.
        print(f"LOGITS SHAPE: {logits}")
        net_output = logits.view(batch_size, n_robots * n_actions)
        return net_output, {}

    # ====================================
    # Custom Value Network (No Masking Needed)
    # ====================================


class SchedulerValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        # For the value function, we simply flatten the observations.
        # Assume the observation dictionary contains "robot_features" and "task_features".
        input_dim = np.prod(observation_space["robot_features"].shape) + np.prod(
            observation_space["task_features"].shape
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        ).to(device)
        self.device = device

    def compute(self, states, role):
        states = unflatten_tensorized_space(self.observation_space, states["states"])
        # Flatten the observation.
        robot_features = states["robot_features"]
        task_features = states["task_features"]
        flat = torch.cat(
            [
                robot_features.view(robot_features.size(0), -1),
                task_features.view(task_features.size(0), -1),
            ],
            dim=1,
        ).to(self.device)
        values = self.net(flat)
        return values, {}


# ============================================
# Set Up Environment, Memory, and PPO Agent
# ============================================

# Register and create your environment.
env_id = "SchedulingRLEnvironment-v0"
gym.register(id=env_id, entry_point="gym_environment_rl:SchedulingRLEnvironment")
env = gym.make(env_id)
state, _ = env.reset()  # state is a dictionary.
env = wrap_env(env)  # SKRL wrapper to provide a PyTorch interface.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a memory buffer.
memory = RandomMemory(memory_size=128, num_envs=1, device=device)

# Instantiate the models.
policy_model = SchedulerPolicy(env.observation_space, env.action_space, device)
value_model = SchedulerValue(env.observation_space, env.action_space, device)
models = {"policy": policy_model, "value": value_model}

# PPO configuration.
ppo_config = PPO_DEFAULT_CONFIG.copy()
ppo_config["rollouts"] = 128
ppo_config["learning_epochs"] = 4
ppo_config["mini_batches"] = 4
ppo_config["discount_factor"] = 0.99
ppo_config["learning_rate"] = 1e-3
ppo_config["experiment"]["write_interval"] = 100

# Instantiate the PPO agent.
agent = PPO(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    cfg=ppo_config,
)

# Setup memory
agent.init()

cfg = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
cfg["headless"] = True  # Disable rendering for training.
trainer = SequentialTrainer(cfg=cfg, env=env, agents=[agent])

trainer.train()

"""
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()  # state is a dictionary.
    done = False
    episode_reward = 0.0
    sim = env.sim
    rl_loop_timetep = 0
    done_cnt = 0
    EXPECTED_TIMESTEPS = 10
    DONE_MINIMUM = 300
    while done_cnt < DONE_MINIMUM:
        # The agent.act() method will call our policy's compute() method,
        # which applies hard masking to ensure only valid actions are sampled.
        actions, log_probs, outputs = agent.act(
            state, timestep=rl_loop_timetep, timesteps=EXPECTED_TIMESTEPS
        )

        print(f"Actions: {actions}")

        robot_assignments = {}
        available_robots = [robot for robot in sim.robots if robot.available]
        available_robot_ids = [robot.robot_id for robot in available_robots]
        incomplete_tasks = [
            task for task in sim.tasks if task.incomplete and task.status == "PENDING"
        ]

        # Check if all normal tasks are done -> send all robots to the exit task
        if len(incomplete_tasks) == 1:  # Only end task incomplete
            for robot in available_robots:
                robot_assignments[robot.robot_id] = incomplete_tasks[0].task_id
                robot.current_task = incomplete_tasks[0]
        else:
            ########################
            robot_reward_matrix = outputs["net_output"].view(env.n_robots, env.n_tasks + 1)
            sim.find_highest_non_idle_reward(robot_reward_matrix)

            # for robot_id in actions:
            # robot_assignments[robot_id] = actions[robot_id]

        # schedule = Instantaneous_Schedule(robot_assignments)

        next_state, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Record transition.
        agent.record_transition(
            states=state,
            actions=actions,
            rewards=reward,
            next_states=next_state,
            terminated=terminated,
            truncated=truncated,
            infos={},
            timestep=rl_loop_timetep,
            timesteps=EXPECTED_TIMESTEPS,
        )
        state = next_state
        episode_reward += reward
        rl_loop_timetep += 1
        done_cnt += 1

    print(f"Episode {episode}: Total reward: {episode_reward}, Stariting UPDATE")
    # After each episode, perform optimization.
    agent._update(timestep=0, timesteps=EXPECTED_TIMESTEPS)
    print(f"Episode {episode}: Reward {episode_reward}")

# Optionally, save the agent checkpoint.
agent.save("ppo_scheduler_agent.pt")
"""
