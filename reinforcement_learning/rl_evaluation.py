import argparse
import sys

sys.path.append("..")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models.policy_value_discrete import SchedulerPolicy
from visualizations.benchmark_visualizations import (
    compare_makespans_1v1,
    plot_violin,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL scheduler using unvectorized gym environment"
    )
    parser.add_argument(
        "--RL_agent_path", type=str, required=True, help="Path to the pretrained model checkpoint"
    )
    parser.add_argument("--n_runs", type=int, default=50, help="Number of runs")
    parser.add_argument(
        "--n_trials_per_run",
        type=int,
        default=10,
        help="Number of trials per run (only used in sampling mode)",
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        choices=["sampling", "argmax"],
        default="sampling",
        help="Policy mode: sampling (multiple trials) or argmax (1 trial)",
    )
    parser.add_argument(
        "--not_use_idle",
        action="store_true",
        default=False,
        help="Use pretrained model",
    )

    args = parser.parse_args()
    use_idle = not args.not_use_idle

    seed = 105
    np.random.seed(seed)
    env_id = "SchedulingRLEnvironment-v0"
    gym.register(
        id=env_id,
        entry_point="gym_environment_rl:SchedulingRLEnvironment",
        kwargs={"use_idle": use_idle},
    )
    env = gym.make(env_id, kwargs={"use_idle": use_idle})
    env_config = env.unwrapped.get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    policy_model = SchedulerPolicy(
        env.observation_space, env.action_space, device, policy_config, use_idle=use_idle
    )
    checkpoint = torch.load(args.RL_agent_path, map_location=device, weights_only=True)
    policy_model.load_state_dict(checkpoint["policy"])
    policy_model.eval().to(device)

    greedy_makespans = []
    sadcher_rl_makespans = []
    trial_count = 1 if args.policy_mode == "argmax" else args.n_trials_per_run
    max_probs_per_run = []

    for run in tqdm(range(args.n_runs)):
        state, info = env.reset()
        greedy_makespans.append(env.unwrapped.return_greedy_makespan())
        best_rl_makespan = float("inf")

        for trial in range(trial_count):
            env.unwrapped.reset_same_problem_instance()
            state, info = env.unwrapped._get_observation(), {}
            done = False

            while not done:
                state_tensor = {
                    k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                    for k, v in state.items()
                }

                action_probas, _ = policy_model.compute(state_tensor, eval=True)
                if use_idle:
                    action_probas = action_probas.reshape(
                        (1, env_config["n_robots"], env_config["n_tasks"] + 1)
                    )
                else:
                    action_probas = action_probas.reshape(
                        (1, env_config["n_robots"], env_config["n_tasks"])
                    )

                probs = action_probas.squeeze(0)
                max_probs_per_run.append(probs.max().item())

                if args.policy_mode == "argmax":
                    action = torch.argmax(probs, dim=1).cpu()
                else:
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().cpu()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state

            current_rl_makespan = env.unwrapped.return_final_makespan()
            best_rl_makespan = min(best_rl_makespan, current_rl_makespan)

        sadcher_rl_makespans.append(best_rl_makespan)

    avg_max_prob = np.mean(max_probs_per_run)
    print(f"\nAverage max action probability across all steps: {avg_max_prob:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    data = {"Greedy": greedy_makespans, "Sadcher-RL": sadcher_rl_makespans}
    scheduler_names = ["Greedy", "Sadcher-RL"]
    plot_violin(
        axs[0],
        data,
        scheduler_names,
        "makespan",
        "Gym Environment RL Benchmark Makespan Distribution",
    )
    compare_makespans_1v1(axs[1], greedy_makespans, sadcher_rl_makespans, "Greedy", "Sadcher-RL")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
