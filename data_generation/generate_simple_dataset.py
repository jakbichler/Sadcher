import json
import os
import argparse
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from problem_generator import ProblemData, generate_simple_data
from baselines.aswale_23.MILP_solver import milp_scheduling


def generate_simple_dataset(n_instances: int, output_dir: str):
    """
    Generates a dataset of problem instances and their optimal solutions.

    Args:
        n_instances (int): Number of problem instances to generate.
        output_dir (str): Directory to save problem instances and solutions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, "problem_instances"))
        os.makedirs(os.path.join(output_dir, "solutions"))

    for instance_index in tqdm(range(n_instances)):
        problem_instance = generate_simple_data()

        # Convert numpy arrays to lists for JSON serialization
        serializable_problem_instance = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in problem_instance.items()
        }


        problem_instance_path = os.path.join(
            output_dir, "problem_instances", f"problem_instance_{instance_index}.json"
        )
        with open(problem_instance_path, "w") as f:
            json.dump(serializable_problem_instance, f)

        # Solve the problem instance using the MILP solver
        optimal_schedule = milp_scheduling(problem_instance)

        solution_path = os.path.join(
            output_dir, "solutions", f"optimal_schedule_{instance_index}.json"
        )
        with open(solution_path, "w") as f:
            json.dump(optimal_schedule.to_dict(), f)


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

    args = parser.parse_args()

    generate_simple_dataset(args.num_instances, args.output_dir)
