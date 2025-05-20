import argparse
import sys
import time

import matplotlib.pyplot as plt
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
from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import Full_Horizon_Schedule, calculate_traveled_distance
from schedulers.initialize_schedulers import create_scheduler
from schedulers.sadcher import SadcherScheduler
from schedulers.sadcherRL import RLSadcherScheduler
from simulation_environment.simulator_2D import Simulation
from visualizations.benchmark_visualizations import (
    compare_makespans_1v1,
    plot_violin,
    print_final_results,
)

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

N_TASKS = 10
N_ROBOTS = 3
N_SKILLS = 3
N_PRECEDENCE = 0
N_STOCHASTIC_RUNS = 10
GRID_SIZE = 100
DURATION_FACTOR = 100 / 5
MAKESPAN_FACTOR = 20
TRAVEL_DISTANCE_FACTOR = 100
SEED = 0
np.random.seed(SEED)

SCHEDULERS_WITH_SAMPLING = {"stochastic_IL_sadcher", "rl_sadcher_sampling"}
IL_checkpoint_path = (
    "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
)
RL_checkpoint_path = "/home/jakob/thesis/reinforcement_learning/archived_runs/revisit_discrete/25-05-16_14-21-09-761174_PPO/checkpoints/best_agent.pt"
CHECKPOINT_HETEROMRTA = "/home/jakob/HeteroMRTA/model/save/checkpoint.pth"

checkpoint_map = {
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


def run_one_simulation(problem_instance, scheduler_name, checkpoint_path, sampling=False):
    worst_case_makespan = np.sum(problem_instance["T_e"]) + np.sum(
        [np.max(problem_instance["T_t"][task]) for task in range(len(problem_instance["T_e"]))]
    )
    # RL models do not use idle, as they allow for direct assignment to tasks that can"t be exectued yet
    use_idle = not scheduler_name.startswith("rl_sadcher")
    sim = Simulation(problem_instance, scheduler_name, use_idle=use_idle)

    scheduler = create_scheduler(
        scheduler_name,
        checkpoint_path,
        duration_normalization=sim.duration_normalization,
        location_normalization=sim.location_normalization,
        stddev=0.5,
    )

    feasible = True
    current_run_computation_times = []
    while not sim.sim_done:
        start_time = time.time()
        filter_triggered = False

        if isinstance(scheduler, SadcherScheduler):
            predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
            sim.find_highest_non_idle_reward(predicted_reward)

        elif isinstance(scheduler, RLSadcherScheduler):
            instantaneous_schedule, filter_triggered = scheduler.calculate_robot_assignment(
                sim, sampling=sampling
            )

        else:
            instantaneous_schedule = scheduler.calculate_robot_assignment(sim)

        current_run_computation_times.append(time.time() - start_time)
        sim.assign_tasks_to_robots(instantaneous_schedule)
        sim.step_until_next_decision_point(filter_triggered=filter_triggered)

        if sim.timestep > worst_case_makespan:
            sim.makespan = worst_case_makespan
            feasible = False
            break

    n_tasks = len(problem_instance["T_e"])
    schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)

    return sim.makespan, feasible, current_run_computation_times, schedule


def evaluate_scheduler_in_simulation(
    scheduler_name, problem_instance, checkpoint_map, n_stochastic_runs=1, sampling=False
):
    all_makespans = []
    all_times = []
    all_feasible = []
    all_distances = []

    for _ in range(n_stochastic_runs):
        makespan, feasible, times, schedule = run_one_simulation(
            problem_instance, scheduler_name, checkpoint_map[scheduler_name], sampling=sampling
        )

        all_makespans.append(makespan)
        all_feasible.append(feasible)
        all_times.extend(times)
        all_distances.append(calculate_traveled_distance(schedule, problem_instance["T_t"]))

    best_run = np.argmin(all_makespans)
    best_makespan = all_makespans[best_run]
    best_distance = all_distances[best_run]
    avg_time = np.mean(all_times)
    feasible = any(all_feasible)

    return (
        best_makespan,
        best_distance,
        avg_time,
        feasible,
    )


def plot_results(
    makespans,
    travel_distances,
    computation_times,
    scheduler_names,
    args,
    n_tasks,
    n_robots,
    n_skills,
    n_precedence,
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))

    plot_violin(
        axs[0],
        makespans,
        scheduler_names,
        "makespan",
        f"Makespan Comparison on {args.n_iterations} instances (seed 0) of {n_tasks}t{n_robots}r{n_skills}s{n_precedence}p",
    )

    plot_violin(
        axs[1],
        computation_times,
        scheduler_names,
        "computation_time",
        "Computation Time Comparison",
    )

    plot_violin(
        axs[2],
        travel_distances,
        scheduler_names,
        "travel_distance",
        "Travel Distance Comparison",
    )

    # compare_makespans_1v1(
    # axs[0, 1], makespans["greedy"], makespans["sadcher"], "Greedy", "Sadcher-RT"
    # )

    # if args.include_milp:
    # compare_makespans_1v1(
    # axs[1, 1], makespans["milp"], makespans["sadcher"], "MILP", "Sadcher-RT"
    # )
    # else:
    # fig.delaxes(axs[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    scheduler_names = ["greedy", "sadcher"]

    if args.include_milp:
        scheduler_names.append("milp")
    if args.include_stochastic_IL_sadcher:
        scheduler_names.append("stochastic_IL_sadcher")
    if args.include_RL_sadcher:
        scheduler_names.extend(["rl_sadcher", "rl_sadcher_sampling"])
    if args.include_heteromrta:
        scheduler_names.extend(["heteromrta", "heteromrta_sampling"])

    preferred_order = [
        "milp",
        "sadcher",
        "stochastic_IL_sadcher",
        "heteromrta",
        "heteromrta_sampling",
        "greedy",
    ]

    scheduler_names = [scheduler for scheduler in preferred_order if scheduler in scheduler_names]

    makespans = {scheduler: [] for scheduler in scheduler_names}
    travel_distances = {scheduler: [] for scheduler in scheduler_names}
    computation_times = {scheduler: [] for scheduler in scheduler_names}
    infeasible_count = {scheduler: 0 for scheduler in scheduler_names}

    for iteration in tqdm(range(args.n_iterations)):
        problem_instance = generate_random_data_with_precedence(
            N_TASKS, N_ROBOTS, N_SKILLS, N_PRECEDENCE
        )

        for scheduler_name in scheduler_names:
            # MILP does not need simulation
            if scheduler_name == "milp":
                start_time = time.time()
                optimal_schedule = milp_scheduling(
                    problem_instance, n_threads=6, cutoff_time_seconds=600
                )
                computation_times[scheduler_name].append(time.time() - start_time)
                makespans[scheduler_name].append(optimal_schedule.makespan)
                travel_distance = calculate_traveled_distance(
                    optimal_schedule, problem_instance["T_t"]
                )
                travel_distances[scheduler_name].append(travel_distance)

            # HeteroMRTA runs
            elif scheduler_name in {"heteromrta", "heteromrta_sampling"}:
                # set up env & model
                EnvParams.TRAIT_DIM = 5
                TrainParams.EMBEDDING_DIM = 128
                TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
                TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

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

                best_ms = None
                runs = 1 if scheduler_name == "heteromrta" else N_STOCHASTIC_RUNS
                for _ in range(runs):
                    env.init_state()
                    worker.env = env
                    if scheduler_name == "heteromrta":
                        _, _, res = worker.run_episode(False, sample=False, max_waiting=False)
                    else:
                        _, _, res_candidate = worker.run_episode(
                            False, sample=True, max_waiting=False
                        )
                        # keep best makespan
                        res = (
                            res_candidate
                            if best_ms is None or res_candidate["makespan"][-1] < best_ms
                            else res
                        )
                    ms = res["makespan"][-1] * MAKESPAN_FACTOR
                    best_ms = ms if best_ms is None else min(best_ms, ms)
                makespans[scheduler_name].append(best_ms)
                travel_distances[scheduler_name].append(
                    res["travel_dist"][-1] * TRAVEL_DISTANCE_FACTOR
                )
                computation_times[scheduler_name].append(np.mean(res["time_per_decision"]))

            # All other schedulers need our  simulation
            else:
                is_stochastic = scheduler_name in SCHEDULERS_WITH_SAMPLING
                n_runs = N_STOCHASTIC_RUNS if is_stochastic else 1
                sampling = is_stochastic

                best_ms, best_dist, avg_time, feasible = evaluate_scheduler_in_simulation(
                    scheduler_name, problem_instance, checkpoint_map, n_runs, sampling
                )

                makespans[scheduler_name].append(best_ms)
                travel_distances[scheduler_name].append(best_dist)
                computation_times[scheduler_name].append(avg_time)
                if not feasible:
                    infeasible_count[scheduler_name] += 1

        iteration_results = sorted((makespans[s][-1], s) for s in scheduler_names)

    print_final_results(
        scheduler_names, args.n_iterations, makespans, infeasible_count, computation_times
    )

    plot_results(
        makespans,
        travel_distances,
        computation_times,
        scheduler_names,
        args,
        N_TASKS,
        N_ROBOTS,
        N_SKILLS,
        N_PRECEDENCE,
    )
