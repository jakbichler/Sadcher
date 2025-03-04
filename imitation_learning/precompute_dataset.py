import os

import torch
from dataset import LazyLoadedSchedulingDataset
from tqdm import tqdm


def precompute_features(problem_dir, solution_dir, output_dir, gamma=0.99, immediate_reward=10):
    os.makedirs(output_dir, exist_ok=True)
    dataset = LazyLoadedSchedulingDataset(
        problem_dir, solution_dir, gamma=gamma, immediate_reward=immediate_reward
    )

    for idx in tqdm(range(len(dataset)), desc="Precomputing"):
        robot_feats, task_feats, expert_reward, feasibility_mask = dataset[idx]
        sample = {
            "robot_feats": robot_feats,
            "task_feats": task_feats,
            "expert_reward": expert_reward,
            "feasibility_mask": feasibility_mask,
        }
        torch.save(sample, os.path.join(output_dir, f"sample_{idx:06d}.pt"))


if __name__ == "__main__":
    # Argument parsing
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--problem_dir", type=str, required=True, help="Directory containing problem instances."
    )
    arg_parser.add_argument(
        "--solution_dir", type=str, required=True, help="Directory containing solution instances."
    )
    arg_parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save precomputed features."
    )
    args = arg_parser.parse_args()

    precompute_features(
        problem_dir=args.problem_dir, solution_dir=args.solution_dir, output_dir=args.output_dir
    )
