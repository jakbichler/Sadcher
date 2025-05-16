import argparse
import json
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append("..")
import os

from benchmark_schedulers import run_one_simulation

from baselines.aswale_23.MILP_solver import milp_scheduling
from data_generation.problem_generator import (
    generate_random_data,
    generate_random_data_with_precedence,
)


def main():
    parser = argparse.ArgumentParser(description="Generate scaling data for scheduling algorithms")
    parser.add_argument("--min_tasks", type=int, required=True, help="Minimum number of tasks")
    parser.add_argument("--max_tasks", type=int, required=True, help="Maximum number of tasks")
    parser.add_argument("--step_tasks", type=int, default=1, help="Step size for tasks")
    parser.add_argument("--min_robots", type=int, required=True, help="Minimum number of robots")
    parser.add_argument("--max_robots", type=int, required=True, help="Maximum number of robots")
    parser.add_argument("--step_robots", type=int, default=1, help="Step size for robots")
    parser.add_argument("--n_skills", type=int, default=2, help="Number of skills")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs per configuration")
    parser.add_argument(
        "--include_milp",
        default=False,
        action="store_true",
        help="Include MILP in the comparison",
    )
    parser.add_argument(
        "--milp_cutoff_time", type=int, default=10 * 60, help="Cutoff time for MILP"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to scheduler checkpoint",
    )
    parser.add_argument("--model_name", type=str, default="8t3r3s", help="Model name for scheduler")
    parser.add_argument(
        "--output_file", type=str, required=True, help="JSON Lines file to append results to"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="random_with_precedence",
        help="Type of problem to generate",
    )
    parser.add_argument(
        "--n_precedence", type=int, default=3, help="Number of precedence constraints"
    )

    parser.add_argument(
        "--include_stochastic_sadcher",
        default=False,
        action="store_true",
        help="Include stochastic Sadcher in the comparison",
    )

    parser.add_argument(
        "--n_stochastic_runs",
        type=int,
        default=10,
        help="Number of stochastic runs for Sadcher",
    )

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scheduler_names = ["greedy", "sadcher", "random_bipartite"]

    if args.include_milp:
        scheduler_names.append("milp")
    if args.include_stochastic_sadcher:
        scheduler_names.append("stochastic_sadcher")

    total_iterations = (
        ((args.max_tasks - args.min_tasks) // args.step_tasks + 1)
        * ((args.max_robots - args.min_robots) // args.step_robots + 1)
        * args.n_runs
    )

    pbar = tqdm(total=total_iterations, desc="Running configurations")
    with open(args.output_file, "a") as outfile:
        for n_tasks in range(args.min_tasks, args.max_tasks + 1, args.step_tasks):
            for n_robots in range(args.min_robots, args.max_robots + 1, args.step_robots):
                for run in range(args.n_runs):
                    run_results = []
                    # Set seed for reproducibility
                    seed = np.random.randint(1, 1e6)
                    np.random.seed(seed)

                    # Generate a random problem instance
                    if args.problem_type == "random":
                        problem_instance = generate_random_data(
                            n_tasks, n_robots, args.n_skills, []
                        )
                    elif args.problem_type == "random_with_precedence":
                        problem_instance = generate_random_data_with_precedence(
                            n_tasks, n_robots, args.n_skills, args.n_precedence
                        )
                    worst_case_makespan = np.sum(problem_instance["T_e"]) + np.sum(
                        [np.max(problem_instance["T_t"][task]) for task in range(n_tasks + 1)]
                    )

                    for scheduler in scheduler_names:
                        if scheduler == "milp":
                            start_time = time.time()
                            optimal_schedule = milp_scheduling(
                                problem_instance, n_threads=6, cutoff_time_seconds=10 * 60
                            )
                            elapsed_time = time.time() - start_time
                            if optimal_schedule is None:
                                makespan = worst_case_makespan
                                infeasible = 1
                            else:
                                makespan = optimal_schedule.makespan
                                infeasible = 0
                            avg_comp_time = elapsed_time
                            total_comp_time = elapsed_time

                        elif scheduler == "stochastic_sadcher":
                            best_ms = float("inf")

                            for _ in range(args.n_stochastic_runs):
                                makespan, feasible, current_run_computation_times = (
                                    run_one_simulation(
                                        problem_instance, scheduler, args.checkpoint_path
                                    )
                                )

                                best_ms = min(best_ms, makespan) if feasible else best_ms

                            if len(current_run_computation_times) > 0:
                                avg_comp_time = float(np.mean(current_run_computation_times))
                                total_comp_time = float(np.sum(current_run_computation_times))
                            else:
                                avg_comp_time = 0.0
                                total_comp_time = 0.0
                            makespan = best_ms
                            infeasible = 1 if best_ms == float("inf") else 0

                        else:
                            makespan, feasible, current_run_computation_times = run_one_simulation(
                                problem_instance, scheduler, args.checkpoint_path
                            )
                            if len(current_run_computation_times) > 0:
                                avg_comp_time = float(np.mean(current_run_computation_times))
                                total_comp_time = float(np.sum(current_run_computation_times))
                            else:
                                avg_comp_time = 0.0
                                total_comp_time = 0.0
                            infeasible = not feasible

                        result = {
                            "n_tasks": n_tasks,
                            "n_robots": n_robots,
                            "n_skills": args.n_skills,
                            "run": seed,
                            "scheduler": scheduler,
                            "makespan": makespan,
                            "avg_comp_time": avg_comp_time,
                            "total_comp_time": total_comp_time,
                            "infeasible_count": infeasible,
                            "model_name": args.model_name,
                            "n_precedence": args.n_precedence,
                        }

                        run_results.append(result)
                    for result in run_results:
                        outfile.write(json.dumps(result) + "\n")
                    outfile.flush()  # Ensure immediate write to disk
                    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
