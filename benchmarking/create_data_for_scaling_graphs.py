import argparse
import os
import signal
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")
import json
import sys

from baselines.aswale_23.MILP_solver import milp_scheduling
from baselines.heteromrta.attention import AttentionNet
from baselines.heteromrta.bridge import problem_to_taskenv
from baselines.heteromrta.env.task_env import TaskEnv
from baselines.heteromrta.parameters import EnvParams, TrainParams
from baselines.heteromrta.worker import Worker
from benchmarking.benchmark_helpers import (
    EpisodeTimeout,
    evaluate_scheduler_in_simulation,
    get_scheduler_names,
    timeout_handler,
)
from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import calculate_traveled_distance
from visualizations.benchmark_visualizations import plot_results, print_final_results

N_SKILLS = 3
N_PRECEDENCE = 3
N_STOCHASTIC_RUNS = 10
GRID_SIZE = 100
DURATION_FACTOR = 100 / 5
MAKESPAN_FACTOR = 20
TRAVEL_DISTANCE_FACTOR = 100
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

SCHEDULERS_WITH_SAMPLING = {"stochastic_IL_sadcher", "rl_sadcher_sampling"}
IL_checkpoint_path = (
    "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
)
RL_checkpoint_path = "/home/jakob/thesis/reinforcement_learning/archived_runs/revisit_discrete/25-05-16_14-21-09-761174_PPO/checkpoints/best_agent.pt"
CHECKPOINT_HETEROMRTA = "/home/jakob/HeteroMRTA/model/save/checkpoint.pth"

CHECKPOINT_MAP = {
    "sadcher": IL_checkpoint_path,
    "stochastic_IL_sadcher": IL_checkpoint_path,
    "rl_sadcher": RL_checkpoint_path,
    "rl_sadcher_sampling": RL_checkpoint_path,
    "greedy": None,
}


def main():
    parser = argparse.ArgumentParser(description="Generate scaling data for scheduling algorithms")
    parser.add_argument("--min_tasks", type=int, required=True, help="Minimum number of tasks")
    parser.add_argument("--max_tasks", type=int, required=True, help="Maximum number of tasks")
    parser.add_argument("--step_tasks", type=int, default=1, help="Step size for tasks")
    parser.add_argument("--min_robots", type=int, required=True, help="Minimum number of robots")
    parser.add_argument("--max_robots", type=int, required=True, help="Maximum number of robots")
    parser.add_argument("--step_robots", type=int, default=1, help="Step size for robots")
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
        "--output_file", type=str, required=True, help="JSON Lines file to append results to"
    )

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scheduler_names = [
        "sadcher",
        "stochastic_IL_sadcher",
        "heteromrta",
        "heteromrta_sampling",
        "greedy",
    ]
    if args.include_milp:
        scheduler_names.append("milp")

    total_iterations = (
        ((args.max_tasks - args.min_tasks) // args.step_tasks + 1)
        * ((args.max_robots - args.min_robots) // args.step_robots + 1)
        * args.n_runs
    )

    pbar = tqdm(total=total_iterations, desc="Running configurations")
    with open(args.output_file, "a") as outfile:
        for run in range(args.n_runs):
            for n_tasks in range(args.min_tasks, args.max_tasks + 1, args.step_tasks):
                for n_robots in range(args.min_robots, args.max_robots + 1, args.step_robots):
                    run_results = []
                    # Set seed for reproducibility
                    seed = np.random.randint(1, 1e6)
                    np.random.seed(seed)

                    problem_instance = generate_random_data_with_precedence(
                        n_tasks, n_robots, N_SKILLS, N_PRECEDENCE
                    )
                    worst_case_makespan = np.sum(problem_instance["T_e"]) + np.sum(
                        [np.max(problem_instance["T_t"][task]) for task in range(n_tasks + 1)]
                    )

                    for scheduler_name in scheduler_names:
                        # MILP does not need simulation
                        if scheduler_name == "milp":
                            start_time = time.time()
                            optimal_schedule = milp_scheduling(
                                problem_instance,
                                n_threads=2,
                                cutoff_time_seconds=args.milp_cutoff_time,
                            )
                            if optimal_schedule is None:
                                print("MILP could not find a solution within time limit.")
                                makespan = worst_case_makespan
                                travel_distance = 0
                                computation_time_per_decision = 0
                                computation_time_full_solution = 0
                                infeasible = True

                            else:
                                makespan = optimal_schedule.makespan
                                infeasible = False
                                travel_distance = calculate_traveled_distance(
                                    optimal_schedule, problem_instance["T_t"]
                                )
                                # MILP can not return intermediate decisions -> both times are the same
                                computation_time_per_decision = time.time() - start_time
                                computation_time_full_solution = time.time() - start_time

                        # HeteroMRTA runs
                        elif scheduler_name in {"heteromrta", "heteromrta_sampling"}:
                            t_start = time.time()
                            # load network once per iteration
                            env: TaskEnv = problem_to_taskenv(
                                problem_instance, GRID_SIZE, DURATION_FACTOR
                            )
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            net = AttentionNet(
                                TrainParams.AGENT_INPUT_DIM,
                                TrainParams.TASK_INPUT_DIM,
                                TrainParams.EMBEDDING_DIM,
                            ).to(device)
                            ckpt = torch.load(CHECKPOINT_HETEROMRTA, map_location=device)
                            net.load_state_dict(ckpt["best_model"])
                            worker = Worker(0, net, net, 0, device)

                            best_ms = float("inf")
                            best_travel_distance = float("inf")
                            best_comp_time_per_decision = float("inf")
                            found_feasible = False
                            sampling_runs = (
                                1 if scheduler_name == "heteromrta" else N_STOCHASTIC_RUNS
                            )

                            for run in range(sampling_runs):
                                env.init_state()
                                worker.env = env
                                try:
                                    signal.signal(signal.SIGALRM, timeout_handler)
                                    signal.alarm(
                                        10
                                    )  # 10s to finish -> catch very rare infinite loops

                                    # run episode (sample only for heteromrta_sampling)
                                    _, _, res_temp = worker.run_episode(
                                        False,
                                        sample=(scheduler_name == "heteromrta_sampling"),
                                        max_waiting=False,
                                    )
                                    signal.alarm(0)  # cancel alarm on success

                                    # compute makespan, clamp by worst_case
                                    ms_temp = res_temp["makespan"][-1] * MAKESPAN_FACTOR
                                    feasible_temp = True
                                    if ms_temp > worst_case_makespan:
                                        ms_temp = worst_case_makespan
                                        feasible_temp = False

                                    # update best
                                    if ms_temp < best_ms:
                                        best_ms = ms_temp
                                        best_travel_distance = (
                                            res_temp["travel_dist"][-1] * TRAVEL_DISTANCE_FACTOR
                                        )
                                        best_comp_time_per_decision = np.mean(
                                            res_temp["time_per_decision"]
                                        )

                                    found_feasible = found_feasible or feasible_temp

                                except Exception as e:
                                    if isinstance(e, EpisodeTimeout):
                                        print(f"  ⚠️  run {run} timed out → marking infeasible")
                                    else:
                                        print(f"  ⚠️  run {run} error ({e!r}) → marking infeasible")

                                    signal.alarm(0)
                                    best_ms = worst_case_makespan
                                    found_feasible = False
                                    break

                            # store metrics for the single best run
                            makespan = best_ms
                            travel_distance = best_travel_distance
                            computation_time_per_decision = best_comp_time_per_decision
                            computation_time_full_solution = time.time() - t_start
                            infeasible = not found_feasible

                        # All other schedulers need our simulation
                        else:
                            is_stochastic = scheduler_name in SCHEDULERS_WITH_SAMPLING
                            n_runs = N_STOCHASTIC_RUNS if is_stochastic else 1
                            sampling = is_stochastic

                            best_ms, best_dist, avg_time, total_time, feasible = (
                                evaluate_scheduler_in_simulation(
                                    scheduler_name,
                                    problem_instance,
                                    CHECKPOINT_MAP,
                                    n_runs,
                                    sampling,
                                    worst_case_makespan,
                                )
                            )

                            makespan = best_ms
                            travel_distance = best_dist
                            computation_time_per_decision = avg_time
                            computation_time_full_solution = total_time
                            infeasible = not feasible

                        result = {
                            "n_tasks": n_tasks,
                            "n_robots": n_robots,
                            "n_skills": N_SKILLS,
                            "n_precedence": N_PRECEDENCE,
                            "run": seed,
                            "scheduler": scheduler_name,
                            "makespan": makespan,
                            "travel_distance": travel_distance,
                            "computation_time_per_decision": computation_time_per_decision,
                            "computation_time_full_solution": computation_time_full_solution,
                            "infeasible_count": infeasible,
                        }

                        run_results.append(result)
                    for result in run_results:
                        outfile.write(json.dumps(result) + "\n")
                    outfile.flush()  # Ensure immediate write to disk
                    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
