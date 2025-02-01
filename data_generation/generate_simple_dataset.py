import json
import os
import argparse
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from problem_generator import ProblemData, generate_simple_data, generate_simple_homogeneous_data, generate_static_data, generate_biased_homogeneous_data
from baselines.aswale_23.MILP_solver import milp_scheduling


def generate_simple_dataset(n_instances: int, output_dir: str, problem_instance_type:str, n_robots = 2, n_tasks = 6) -> None:
    """
    Generates a dataset of problem instances and their optimal solutions.

    Args:
        n_instances (int): Number of problem instances to generate.
        output_dir (str): Directory to save problem instances and solutions.
        type (str): Type of problem instances to generate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, "problem_instances"))
        os.makedirs(os.path.join(output_dir, "solutions"))

    for instance_index in tqdm(range(n_instances)):
        
        if problem_instance_type == "heterogeneous":
            problem_instance = generate_simple_data()
        elif problem_instance_type == "homogeneous":
            problem_instance = generate_simple_homogeneous_data(n_tasks=n_tasks, n_robots=n_robots)  
        elif problem_instance_type == "biased_homogeneous":
            problem_instance = generate_biased_homogeneous_data()
        elif problem_instance_type == "static":
            problem_instance = generate_static_data()

        else:
            raise ValueError(f"Invalid problem instance type: {problem_instance_type}")
        

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
        optimal_schedule = milp_scheduling(problem_instance, n_threads=8)

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

    args = parser.parse_args()

    generate_simple_dataset(args.num_instances, args.output_dir, problem_instance_type=args.problem_instance_type, 
                            n_robots=args.n_robots, n_tasks=args.n_tasks)
