import json
import os
import numpy as np

from MILP_solver import milp_scheduling
from problem_generator import generate_random_data, generate_random_data_with_precedence


def generate_dataset(n_instances: int, output_dir: str, n_robots=2, n_tasks=6, n_skills=2, n_threads=6, n_precedence=0) -> None:
    problem_instances_dir = os.path.join(output_dir, "problem_instances")
    solutions_dir = os.path.join(output_dir, "solutions")

    os.makedirs(problem_instances_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)

    successful = 0


    while successful < n_instances:
        #problem_instance = generate_random_data(n_tasks=n_tasks, n_robots=n_robots, n_skills=n_skills)
        problem_instance = generate_random_data_with_precedence(n_tasks, n_robots, n_skills, n_precedence)
        optimal_schedule = milp_scheduling(problem_instance, n_threads=n_threads, cutoff_time_seconds= 60*10)

        if optimal_schedule is None:
            print(f"Failed to solve problem instance at index {instance_index}. Retrying with a new instance...")
            continue  # Do not increment instance_index; try again

        # Prepare instance for JSON serialization
        serializable_problem_instance = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in problem_instance.items()
        }
        

        instance_index = get_next_available_index(output_dir)

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

        successful += 1
        print(f"Wrote {instance_index} successfully")



def get_next_available_index(output_dir: str) -> int:
    """
    Determines the next available problem instance index by checking existing files.
    """
    problem_instance_dir = os.path.join(output_dir, "problem_instances")
    
    if not os.path.exists(problem_instance_dir):
        return 0  # No existing files, start from 0

    existing_files = [f for f in os.listdir(problem_instance_dir) if f.startswith("problem_instance_")]
    existing_indices = [
        int(f.replace("problem_instance_", "").replace(".json", "")) 
        for f in existing_files if f.replace("problem_instance_", "").replace(".json", "").isdigit()
    ]

    return max(existing_indices, default=-1) + 1  # Start from next available index



if __name__ == "__main__":

    num_instances = 100
    output_dir = "/scratch/jbichler/generate_mrta_datasets/10t3r3s3p"
    n_tasks = 10
    n_robots = 3
    n_skills = 3
    n_threads = 1
    n_precedence = 3

    generate_dataset(num_instances, output_dir, n_robots, n_tasks, n_skills, n_threads, n_precedence)
