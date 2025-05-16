import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append("..")
from baselines.aswale_23.MILP_solver import milp_scheduling
from data_generation.problem_generator import generate_random_data_with_precedence
from schedulers.initialize_schedulers import create_scheduler
from schedulers.sadcher import SadcherScheduler
from schedulers.sadcherRL import RLSadcherScheduler
from simulation_environment.simulator_2D import Simulation
from visualizations.benchmark_visualizations import (
    compare_makespans_1v1,
    plot_violin,
    print_final_results,
)

N_TASKS = 10
N_ROBOTS = 3
N_SKILLS = 3
N_PRECEDENCE = 3
N_STOCHASTIC_RUNS = 20
SEED = 0
np.random.seed(SEED)

SCHEDULERS_WITH_SAMPLING = {"stochastic_IL_sadcher", "rl_sadcher_sampling"}
IL_checkpoint_path = (
    "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
)
RL_checkpoint_path = "/home/jakob/thesis/reinforcement_learning/archived_runs/revisit_discrete/25-05-15_all_instances_frozen_encoders/checkpoints/best_agent.pt"

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

    return sim.makespan, feasible, current_run_computation_times


def evaluate_scheduler_in_simulation(
    scheduler_name, problem_instance, checkpoint_map, n_stochastic_runs=1, sampling=False
):
    all_makespans = []
    all_times = []
    all_feasible = []

    for _ in range(n_stochastic_runs):
        makespan, feasible, times = run_one_simulation(
            problem_instance, scheduler_name, checkpoint_map[scheduler_name], sampling=sampling
        )
        all_makespans.append(makespan)
        all_feasible.append(feasible)
        all_times.extend(times)

    best_ms = min(all_makespans)
    feasible = any(all_feasible)
    avg_time = np.mean(all_times)

    return best_ms, avg_time, feasible


def plot_results(
    makespans, computation_times, scheduler_names, args, n_tasks, n_robots, n_skills, n_precedence
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plot_violin(
        axs[0, 0],
        makespans,
        scheduler_names,
        "makespan",
        f"Makespan Comparison on {args.n_iterations} instances (seed 0) of {n_tasks}t{n_robots}r{n_skills}s{n_precedence}p",
    )

    plot_violin(
        axs[1, 0],
        computation_times,
        scheduler_names,
        "computation_time",
        "Computation Time Comparison",
    )

    compare_makespans_1v1(
        axs[0, 1], makespans["greedy"], makespans["sadcher"], "Greedy", "Sadcher-RT"
    )

    if args.include_milp:
        compare_makespans_1v1(
            axs[1, 1], makespans["milp"], makespans["sadcher"], "MILP", "Sadcher-RT"
        )
    else:
        fig.delaxes(axs[1, 1])

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

    makespans = {scheduler: [] for scheduler in scheduler_names}
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

            # All other schedulers need simulation
            else:
                is_stochastic = scheduler_name in SCHEDULERS_WITH_SAMPLING
                n_runs = N_STOCHASTIC_RUNS if is_stochastic else 1
                sampling = is_stochastic

                best_ms, avg_time, feasible = evaluate_scheduler_in_simulation(
                    scheduler_name, problem_instance, checkpoint_map, n_runs, sampling
                )

                makespans[scheduler_name].append(best_ms)
                computation_times[scheduler_name].append(avg_time)
                if not feasible:
                    infeasible_count[scheduler_name] += 1

        iteration_results = sorted((makespans[s][-1], s) for s in scheduler_names)

    print_final_results(
        scheduler_names, args.n_iterations, makespans, infeasible_count, computation_times
    )

    plot_results(
        makespans,
        computation_times,
        scheduler_names,
        args,
        N_TASKS,
        N_ROBOTS,
        N_SKILLS,
        N_PRECEDENCE,
    )
