import argparse
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")
from baselines.aswale_23.MILP_solver import milp_scheduling
from baselines.heteromrta.attention import AttentionNet
from baselines.heteromrta.bridge import problem_to_taskenv
from baselines.heteromrta.env.task_env import TaskEnv
from baselines.heteromrta.parameters import EnvParams, TrainParams
from baselines.heteromrta.worker import Worker
from benchmarking.benchmark_helpers import evaluate_scheduler_in_simulation, get_scheduler_names
from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import calculate_traveled_distance
from visualizations.benchmark_visualizations import plot_results, print_final_results

"""
Scaling between Sadcher and HeteroMRTA (https://github.com/marmotlab/HeteroMRTA):

– In Sadcher, tasks live on a [0,100]×[0,100] grid with robot speed = 1 unit/timestep.
– In HeteroMRTA, tasks live on a [0,1]×[0,1] grid with robot speed = 0.2 units/timestep.

Since 1 unit in the HeteroMRTA grid corresponds to 100 units in Sadcher’s grid, 
and the HeteroMRTA robot moves at 0.2 versus Sadcher’s speed of 1, 
the effective traversal rate in HeteroMRTA is 0.2×100 = 20× faster 
than in Sadcher.

– In Sadcher, durations are in [50,100] units 
– In HeteroMRTA in [0,5]

So the duration factor is again 20x. 

For fair comparison, we first scale all inputs in the bridge between sadcher problem instance and 
HeteroMRTA environment to match the distribution that HeteroMRTA was trained on.
Afterwards, we multiply all HeteroMRTA makespans by 20 to compare the values.
This results in the exact fair comparison (can be verified: instances with same schedule have same makespan).
"""

N_TASKS = 8
N_ROBOTS = 3
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

SEED = 0
np.random.seed(SEED)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_milp", action="store_true", default=False)
    parser.add_argument("--include_stochastic_IL_sadcher", action="store_true", default=False)
    parser.add_argument("--include_RL_sadcher", action="store_true", default=False)
    parser.add_argument("--include_heteromrta", action="store_true", default=False)
    parser.add_argument("--n_iterations", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.include_milp and N_TASKS >= 10:
        print(
            f"⚠️  Warning: You enabled MILP and N_TASKS={N_TASKS} ≥10. "
            "MILP solve times may grow exponentially and take a very long time."
        )

    scheduler_names = get_scheduler_names(args)
    makespans = {scheduler: [] for scheduler in scheduler_names}
    travel_distances = {scheduler: [] for scheduler in scheduler_names}
    computation_times_per_decision = {scheduler: [] for scheduler in scheduler_names}
    computation_times_full_solution = {scheduler: [] for scheduler in scheduler_names}
    feasibility = {scheduler: [] for scheduler in scheduler_names}

    for iteration in tqdm(range(args.n_iterations)):
        problem_instance = generate_random_data_with_precedence(
            N_TASKS, N_ROBOTS, N_SKILLS, N_PRECEDENCE
        )
        worst_case_makespan = np.sum(problem_instance["T_e"]) + np.sum(
            [np.max(problem_instance["T_t"][task]) for task in range(len(problem_instance["T_e"]))]
        )

        for scheduler_name in scheduler_names:
            # MILP does not need simulation
            if scheduler_name == "milp":
                start_time = time.time()
                optimal_schedule = milp_scheduling(
                    problem_instance, n_threads=12, cutoff_time_seconds=60 * 15
                )
                if optimal_schedule is None:
                    print("MILP could not find a solution within time limit.")
                    makespans[scheduler_name].append(worst_case_makespan)
                    travel_distances[scheduler_name].append(0)
                    computation_times_per_decision[scheduler_name].append(0)
                    computation_times_full_solution[scheduler_name].append(0)
                    feasibility[scheduler_name].append(False)
                    continue

                time_full_solution = time.time() - start_time
                # MILP can not return intermediate decisions -> both times are the same
                computation_times_per_decision[scheduler_name].append(time_full_solution)
                computation_times_full_solution[scheduler_name].append(time_full_solution)
                makespans[scheduler_name].append(optimal_schedule.makespan)
                travel_distance = calculate_traveled_distance(
                    optimal_schedule, problem_instance["T_t"]
                )
                travel_distances[scheduler_name].append(travel_distance)
                feasibility[scheduler_name].append(True)

            # HeteroMRTA runs
            elif scheduler_name in {"heteromrta", "heteromrta_sampling"}:
                t_start = time.time()
                # load network once per iteration
                env: TaskEnv = problem_to_taskenv(problem_instance, GRID_SIZE, DURATION_FACTOR)
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
                best_res = None
                found_feasible = False
                sampling_runs = 1 if scheduler_name == "heteromrta" else N_STOCHASTIC_RUNS

                for _ in range(sampling_runs):
                    env.init_state()
                    worker.env = env
                    try:
                        # run episode (sample only for heteromrta_sampling)
                        _, _, res_temp = worker.run_episode(
                            False,
                            sample=(scheduler_name == "heteromrta_sampling"),
                            max_waiting=False,
                        )

                        # compute makespan, clamp by worst_case
                        ms_temp = res_temp["makespan"][-1] * MAKESPAN_FACTOR
                        feasible_temp = True
                        if ms_temp > worst_case_makespan:
                            ms_temp = worst_case_makespan
                            feasible_temp = False

                        # update best
                        if ms_temp < best_ms:
                            best_ms = ms_temp
                            best_res = res_temp
                        found_feasible = found_feasible or feasible_temp

                        # treat failed run as infeasible

                    except Exception as e:
                        print(f"Error in HeteroMRTA run: {e}")
                        found_feasible = False
                        best_ms = worst_case_makespan
                        break

                # store metrics for the single best run
                makespans[scheduler_name].append(best_ms)
                travel_distances[scheduler_name].append(
                    best_res["travel_dist"][-1] * TRAVEL_DISTANCE_FACTOR
                )
                computation_times_per_decision[scheduler_name].append(
                    np.mean(best_res["time_per_decision"])
                )
                computation_times_full_solution[scheduler_name].append(time.time() - t_start)
                feasibility[scheduler_name].append(found_feasible)

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

                makespans[scheduler_name].append(best_ms)
                travel_distances[scheduler_name].append(best_dist)
                computation_times_per_decision[scheduler_name].append(avg_time)
                computation_times_full_solution[scheduler_name].append(total_time)
                feasibility[scheduler_name].append(feasible)

        iteration_results = sorted((makespans[s][-1], s) for s in scheduler_names)

        with open(f"benchmark_results_{SEED}.pkl", "wb") as f:
            pickle.dump(
                {
                    "makespans": makespans,
                    "travel_distances": travel_distances,
                    "computation_times_per_decision": computation_times_per_decision,
                    "computation_times_full_solution": computation_times_full_solution,
                    "feasibility": feasibility,
                    "scheduler_names": scheduler_names,
                },
                f,
            )

    print_final_results(
        scheduler_names,
        args.n_iterations,
        makespans,
        feasibility,
        computation_times_per_decision,
    )

    plot_results(
        makespans,
        travel_distances,
        computation_times_per_decision,
        computation_times_full_solution,
        feasibility,
        scheduler_names,
        args.n_iterations,
        N_TASKS,
        N_ROBOTS,
        N_SKILLS,
        N_PRECEDENCE,
    )
