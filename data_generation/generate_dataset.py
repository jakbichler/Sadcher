import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from problem_generator import (
    generate_biased_homogeneous_data,
    generate_heterogeneous_no_coalition_data,
    generate_idle_data,
    generate_random_data,
    generate_random_data_with_precedence,
    generate_simple_data,
    generate_simple_homogeneous_data,
    generate_static_data,
)
from tqdm import tqdm

from baselines.aswale_23.MILP_solver import milp_scheduling


def generate_dataset(
    n_instances: int,
    output_dir: str,
    problem_instance_type: str,
    n_robots=2,
    n_tasks=6,
    n_skills=1,
    n_precedence=0,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, "problem_instances"))
        os.makedirs(os.path.join(output_dir, "solutions"))
    else:
        if not os.path.exists(os.path.join(output_dir, "problem_instances")):
            os.makedirs(os.path.join(output_dir, "problem_instances"))
        if not os.path.exists(os.path.join(output_dir, "solutions")):
            os.makedirs(os.path.join(output_dir, "solutions"))

    instance_index = get_next_available_index(output_dir)
    successful = 0

    with tqdm(total=n_instances) as pbar:
        while successful < n_instances:
            if problem_instance_type == "random":
                problem_instance = generate_random_data(
                    n_tasks=n_tasks, n_robots=n_robots, n_skills=n_skills
                )
            elif problem_instance_type == "random_with_precedence":
                problem_instance = generate_random_data_with_precedence(
                    n_tasks=n_tasks, n_robots=n_robots, n_skills=n_skills, n_precedence=n_precedence
                )
            elif problem_instance_type == "heterogeneous":
                problem_instance = generate_simple_data()
            elif problem_instance_type == "heterogeneous_no_coalition":
                problem_instance = generate_heterogeneous_no_coalition_data()
            elif problem_instance_type == "homogeneous":
                problem_instance = generate_simple_homogeneous_data(
                    n_tasks=n_tasks, n_robots=n_robots
                )
            elif problem_instance_type == "biased_homogeneous":
                problem_instance = generate_biased_homogeneous_data()
            elif problem_instance_type == "static":
                problem_instance = generate_static_data()
            elif problem_instance_type == "idle":
                problem_instance = generate_idle_data()
            else:
                raise ValueError(f"Invalid problem instance type: {problem_instance_type}")

            optimal_schedule = milp_scheduling(
                problem_instance, n_threads=6, cutoff_time_seconds=60 * 10
            )

            if optimal_schedule is None:
                print(
                    f"Failed to solve problem instance at index {instance_index}. Retrying with a new instance..."
                )
                continue  # Do not increment instance_index; try again

            # Prepare instance for JSON serialization
            serializable_problem_instance = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in problem_instance.items()
            }
            problem_instance_path = os.path.join(
                output_dir, "problem_instances", f"problem_instance_{instance_index:06d}.json"
            )
            with open(problem_instance_path, "w") as f:
                json.dump(serializable_problem_instance, f)

            solution_path = os.path.join(
                output_dir, "solutions", f"optimal_schedule_{instance_index:06d}.json"
            )
            with open(solution_path, "w") as f:
                json.dump(optimal_schedule.to_dict(), f)

            instance_index += 1  # Increase index only when a valid solution is obtained
            successful += 1
            pbar.update(1)


def get_next_available_index(output_dir: str) -> int:
    """
    Determines the next available problem instance index by checking existing files.
    """
    problem_instance_dir = os.path.join(output_dir, "problem_instances")

    if not os.path.exists(problem_instance_dir):
        return 0  # No existing files, start from 0

    existing_files = [
        f for f in os.listdir(problem_instance_dir) if f.startswith("problem_instance_")
    ]
    existing_indices = [
        int(f.replace("problem_instance_", "").replace(".json", ""))
        for f in existing_files
        if f.replace("problem_instance_", "").replace(".json", "").isdigit()
    ]

    return max(existing_indices, default=-1) + 1  # Start from next available index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset of problem instances and their MILP solutions."
    )
    parser.add_argument(
        "--num_instances",
        "-n",
        type=int,
        required=True,
        help="Number of problem instances to generate.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory to save problem instances and solutions.",
    )

    parser.add_argument(
        "--problem_instance_type",
        "-t",
        type=str,
        required=True,
        help="Type of problem instances to generate: 'heterogeneous' or 'homogeneous'.",
    )

    parser.add_argument(
        "--n_robots",
        type=int,
        required=False,
        help="Number of robots in the problem instance.",
    )

    parser.add_argument(
        "--n_tasks",
        type=int,
        required=False,
        help="Number of tasks in the problem instance.",
    )
    parser.add_argument(
        "--n_skills",
        type=int,
        required=False,
        help="Number of skills in the problem instance.",
    )

    parser.add_argument(
        "--n_precedence",
        type=int,
        required=False,
        help="Number of precedence constraints in the problem instance.",
    )

    args = parser.parse_args()

    generate_dataset(
        args.num_instances,
        args.output_dir,
        problem_instance_type=args.problem_instance_type,
        n_robots=args.n_robots,
        n_tasks=args.n_tasks,
        n_skills=args.n_skills,
        n_precedence=args.n_precedence,
    )
