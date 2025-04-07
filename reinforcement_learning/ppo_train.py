import gymnasium as gym
import torch
from policy_value import SchedulerPolicy, SchedulerValue
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.parallel import PARALLEL_TRAINER_DEFAULT_CONFIG, ParallelTrainer

if __name__ == "__main__":
    N_ENVS = 64

    env_id = "SchedulingRLEnvironment-v0"
    gym.register(id=env_id, entry_point="gym_environment_rl:SchedulingRLEnvironment")
    envs = gym.make_vec(env_id, num_envs=N_ENVS, vectorization_mode="async")
    envs = wrap_env(envs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory = RandomMemory(memory_size=128, num_envs=N_ENVS, device=device)

    policy_model = SchedulerPolicy(envs.observation_space, envs.action_space, device)
    value_model = SchedulerValue(envs.observation_space, envs.action_space)
    models = {"policy": policy_model, "value": value_model}

    ppo_config = PPO_DEFAULT_CONFIG.copy()
    ppo_config["rollouts"] = 256
    ppo_config["learning_epochs"] = 4
    ppo_config["mini_batches"] = 4
    ppo_config["discount_factor"] = 0.99
    ppo_config["learning_rate"] = 3e-4
    ppo_config["experiment"]["write_interval"] = 1000

    agent = PPO(
        models=models,
        memory=memory,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        cfg=ppo_config,
    )
    agent.init()

    cfg = PARALLEL_TRAINER_DEFAULT_CONFIG.copy()
    cfg["headless"] = True
    trainer = ParallelTrainer(cfg=cfg, env=envs, agents=[agent])
    trainer.train()
