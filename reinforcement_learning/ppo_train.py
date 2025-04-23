import argparse
import os
import sys

sys.path.append("..")

import gymnasium as gym
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.parallel import PARALLEL_TRAINER_DEFAULT_CONFIG, ParallelTrainer

from models.policy_value_continuous import ContinuousSchedulerPolicy, ZeroCritic
from models.policy_value_discrete import SchedulerPolicy, SchedulerValue


def write_config(
    policy_config, value_config, ppo_config, trainer_cfg, var, experiment_dir, env_config
):
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = os.path.join(experiment_dir, "config.txt")

    def write_section(f, title, config):
        f.write(f"\n{title}:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    with open(config_path, "w") as f:
        write_section(f, "Command Line Variables", var)
        write_section(f, "Policy Config", policy_config)
        write_section(f, "Value Config", value_config)
        write_section(f, "PPO Configuration", ppo_config)
        write_section(f, "Trainer Configuration", trainer_cfg)
        write_section(f, "Environment Configuration", env_config)


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
        "--IL_pretrained_policy",
        action="store_true",
        default=False,
        help="Use pretrained model",
    )

    argument_parser.add_argument(
        "--RL_pretrained",
        action="store_true",
        default=False,
        help="Use pretrained model",
    )

    argument_parser.add_argument(
        "--RL_pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained model",
    )

    argument_parser.add_argument(
        "--problem_type",
        type=str,
        default="random_with_precedence",
        help="Type of scheduling problem (e.g., 'random_with_precedence' or 'random_all_robots_all_skills')",
    )

    argument_parser.add_argument(
        "--not_use_idle",
        action="store_true",
        default=False,
        help="Use pretrained model",
    )

    argument_parser.add_argument(
        "--continuous_RL",
        action="store_true",
        default=False,
        help="Use continuous RL",
    )

    argument_parser.add_argument(
        "--zero_critic",
        action="store_true",
        default=False,
        help="Use zero critic instead of a learned one",
    )

    args = argument_parser.parse_args()
    use_idle = not args.not_use_idle

    if args.continuous_RL:
        env_id = "SchedulingRLEnvironmentContinuous-v0"

        gym.register(
            id=env_id,
            entry_point="continuous_gym_environment_rl:ContinuousSchedulingRLEnvironment",
            kwargs={
                "problem_type": args.problem_type,
                "use_idle": use_idle,
                "subtractive_assignment": False,
            },
        )
    else:
        env_id = "SchedulingRLEnvironmentDiscrete-v0"
        gym.register(
            id=env_id,
            entry_point="discrete_gym_environment_rl:SchedulingRLEnvironment",
            kwargs={
                "problem_type": args.problem_type,
                "use_idle": use_idle,
                "subtractive_assignment": True,
            },
        )
    envs = gym.make_vec(
        env_id,
        num_envs=args.N_ENVS,
        vectorization_mode="async",
        kwargs={
            "problem_type": args.problem_type,
            "use_idle": use_idle,
            "subtractive_assignment": True,
        },
    )
    env_configs = envs.call("get_config")
    first_env_config = env_configs[0]

    envs = wrap_env(envs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug = args.N_ENVS == 1

    policy_config = {
        "robot_input_dimensions": 7,
        "task_input_dimension": 9,
        "embed_dim": 256,
        "ff_dim": 512,
        "n_transformer_heads": 4,
        "n_transformer_layers": 2,
        "n_gatn_heads": 8,
        "n_gatn_layers": 1,
    }

    value_config = {
        "robot_input_dim": 7,
        "task_input_dim": 9,
        "embed_dim": 128,
        "ff_dim": 256,
        "n_transformer_heads": 2,
        "n_transformer_layers": 1,
        "n_gatn_heads": 1,
        "n_gatn_layers": 1,
    }

    trainer_cfg = PARALLEL_TRAINER_DEFAULT_CONFIG.copy()
    trainer_cfg["timesteps"] = 1_000_000
    trainer_cfg["headless"] = not debug
    trainer_cfg["idle_task_id"] = first_env_config["n_tasks"]

    ppo_config = PPO_DEFAULT_CONFIG.copy()
    ppo_config["rollouts"] = args.N_ROLLOUTS
    ppo_config["learning_epochs"] = 6
    ppo_config["mini_batches"] = 16
    ppo_config["discount_factor"] = 0.99
    ppo_config["learning_rate"] = 2e-4
    ppo_config["mixed_precision"] = False
    ppo_config["experiment"]["write_interval"] = args.N_ROLLOUTS
    ppo_config["experiment"]["checkpoint_interval"] = 2_000
    ppo_config["kl_threshold"] = 0.03
    ppo_config["clip_ratio"] = 0.1
    ppo_config["entropy_loss_scale"] = 0.001
    if args.zero_critic:
        ppo_config["value_loss_scale"] = 0.0
    else:
        ppo_config["value_loss_scale"] = 1.0

    log_stddev_init = 0

    memory = RandomMemory(memory_size=args.N_ROLLOUTS, num_envs=args.N_ENVS, device=device)

    if args.continuous_RL:
        policy_model = ContinuousSchedulerPolicy(
            envs.observation_space,
            envs.action_space,
            device,
            policy_config=policy_config,
            pretrained=args.IL_pretrained_policy,
            use_idle=use_idle,
            debug=debug,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            log_stddev_init=log_stddev_init,
        )

    else:
        policy_model = SchedulerPolicy(
            envs.observation_space,
            envs.action_space,
            device,
            policy_config=policy_config,
            pretrained=args.IL_pretrained_policy,
            use_idle=use_idle,
            use_positional_encoding=False,
            debug=debug,
        )

    print(type(policy_model))

    if args.zero_critic:
        value_model = ZeroCritic(
            envs.observation_space,
            envs.action_space,
            device,
        )
    else:
        value_model = SchedulerValue(envs.observation_space, envs.action_space, value_config).to(
            device
        )

    agent = PPO(
        models={"policy": policy_model, "value": value_model},
        memory=memory,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        cfg=ppo_config,
    )

    if args.RL_pretrained:
        agent.load(path=args.RL_pretrained_path)

    write_config(
        policy_config,
        value_config,
        ppo_config,
        trainer_cfg,
        vars(args),
        agent.experiment_dir,
        first_env_config,
    )

    trainer = ParallelTrainer(cfg=trainer_cfg, env=envs, agents=[agent])
    trainer.train()
