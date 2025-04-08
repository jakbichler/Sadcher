import argparse
import os

import gymnasium as gym
import torch
from policy_value import SchedulerPolicy, SchedulerValue
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.parallel import PARALLEL_TRAINER_DEFAULT_CONFIG, ParallelTrainer


def write_config(ppo_config, trainer_cfg, var, experiment_dir, env_config):
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = os.path.join(experiment_dir, "config.txt")

    with open(config_path, "w") as f:
        f.write("CommandLine Variables:\n")
        for key, value in var.items():
            f.write(f"{key}: {value}\n")
        f.write("\nPPO Configuration:\n")
        for key, value in ppo_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTrainer Configuration:\n")
        for key, value in trainer_cfg.items():
            f.write(f"{key}: {value}\n")
        f.write("\nEnvironment Configuration:\n")
        for key, value in env_config.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Train the model using PPO implemenation of SK RL"
    )

    argument_parser.add_argument(
        "--N_ENVS",
        type=int,
        default=32,
        help="Number of parallel environments",
    )

    argument_parser.add_argument(
        "--N_ROLLOUTS",
        type=int,
        default=256,
        help="Number of rollouts enviroment steps before PPO update",
    )

    argument_parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use pretrained model",
    )

    args = argument_parser.parse_args()
    env_id = "SchedulingRLEnvironment-v0"
    gym.register(id=env_id, entry_point="gym_environment_rl:SchedulingRLEnvironment")
    envs = gym.make_vec(env_id, num_envs=args.N_ENVS, vectorization_mode="async")
    env_configs = envs.call("get_config")
    first_env_config = env_configs[0]

    envs = wrap_env(envs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory = RandomMemory(memory_size=args.N_ROLLOUTS, num_envs=args.N_ENVS, device=device)

    ppo_config = PPO_DEFAULT_CONFIG.copy()
    ppo_config["rollouts"] = args.N_ROLLOUTS
    ppo_config["learning_epochs"] = 4
    ppo_config["mini_batches"] = 4
    ppo_config["discount_factor"] = 0.99
    ppo_config["learning_rate"] = 3e-4
    ppo_config["experiment"]["write_interval"] = 100

    trainer_cfg = PARALLEL_TRAINER_DEFAULT_CONFIG.copy()
    trainer_cfg["timesteps"] = 100_000
    trainer_cfg["headless"] = True
    trainer_cfg["idle_task_id"] = 8

    policy_model = SchedulerPolicy(
        envs.observation_space, envs.action_space, device, pretrained=args.pretrained
    )
    value_model = SchedulerValue(envs.observation_space, envs.action_space)

    agent = PPO(
        models={"policy": policy_model, "value": value_model},
        memory=memory,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        cfg=ppo_config,
    )

    write_config(ppo_config, trainer_cfg, vars(args), agent.experiment_dir, first_env_config)

    trainer = ParallelTrainer(cfg=trainer_cfg, env=envs, agents=[agent])
    trainer.train()
