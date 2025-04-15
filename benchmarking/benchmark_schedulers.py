import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append("..")
from baselines.aswale_23.MILP_solver import milp_scheduling
from data_generation.problem_generator import (
    generate_random_data,
    generate_random_data_with_precedence,
)
from schedulers.sadcher import SadcherScheduler
from simulation_environment.simulator_2D import Simulation, create_scheduler
from visualizations.benchmark_visualizations import (
    compare_makespans_1v1,
    plot_violin,
    print_final_results,
)

if __name__ == "__main__":
    n_tasks = 8
    n_robots = 3
    n_skills = 3
    n_precedence = 3
    seed = 5367
    np.random.seed(seed)
    model_name = "8t3r3s"
    checkpoint_path = (
        "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--including_milp",
        default=False,
        action="store_true",
        help="Include MILP in the comparison",
    )
    arg_parser.add_argument(
        "--n_iterations", type=int, default=50, help="Number of iterations to run"
    )
    arg_parser.add_argument(
        "--move_while_waiting",
        default=False,
        action="store_true",
        help="Allow robots to move while waiting for tasks",
    )
    args = arg_parser.parse_args()

    if args.including_milp:
        scheduler_names = ["milp", "greedy", "sadcher", "random_bipartite"]
    else:
        scheduler_names = ["greedy", "sadcher", "random_bipartite"]

    makespans = {scheduler: [] for scheduler in scheduler_names}
    feasible_makespans = {scheduler: [] for scheduler in scheduler_names}
    computation_times = {scheduler: [] for scheduler in scheduler_names}
    infeasible_count = {scheduler: 0 for scheduler in scheduler_names}

    for iteration in tqdm(range(args.n_iterations)):
        # Generate a problem instance
        # problem_instance = generate_random_data(n_tasks, n_robots, n_skills, [])
        print(iteration)
        problem_instance = generate_random_data_with_precedence(
            n_tasks, n_robots, n_skills, n_precedence
        )

        worst_case_makespan = np.sum(problem_instance["T_e"]) + np.sum(
            [np.max(problem_instance["T_t"][task]) for task in range(n_tasks + 1)]
        )

        for scheduler_name in scheduler_names:
            if scheduler_name == "milp":
                start_time = time.time()
                optimal_schedule = milp_scheduling(
                    problem_instance, n_threads=6, cutoff_time_seconds=10 * 60
                )
                computation_times[scheduler_name].append(time.time() - start_time)
                makespans[scheduler_name].append(optimal_schedule.makespan)
                feasible_makespans[scheduler_name].append(optimal_schedule.makespan)
            else:
                # Non-MILP olutins need simulation to roll out schedule
                sim = Simulation(problem_instance, scheduler_name)
                scheduler = create_scheduler(
                    scheduler_name,
                    checkpoint_path,
                    model_name="8t3r3s",
                    duration_normalization=sim.duration_normalization,
                    location_normalization=sim.location_normalization,
                )
                feasible = True
                current_run_computation_times = []
                while not sim.sim_done:
                    start_time = time.time()
                    if isinstance(scheduler, SadcherScheduler):
                        predicted_reward, instantaneous_schedule = (
                            scheduler.calculate_robot_assignment(sim)
                        )
                        sim.find_highest_non_idle_reward(predicted_reward)
                    else:
                        instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
                    current_run_computation_times.append(time.time() - start_time)

                    sim.assign_tasks_to_robots(instantaneous_schedule)
                    sim.step_until_next_decision_point()

                    if sim.timestep > worst_case_makespan:
                        sim.makespan = worst_case_makespan  # No feasible solution found
                        infeasible_count[scheduler_name] += 1
                        print(
                            f"Scheduler {scheduler_name} did not find a feasible solution at {iteration}"
                        )
                        feasible = False
                        break

                makespans[scheduler_name].append(sim.makespan)
                average_computation_time_per_assignment = np.mean(current_run_computation_times)
                full_computation_time = np.sum(current_run_computation_times)
                computation_times[scheduler_name].append(average_computation_time_per_assignment)

                if feasible:
                    feasible_makespans[scheduler_name].append(sim.makespan)

        iteration_results = sorted(
            (makespans[scheduler_name][-1], scheduler_name) for scheduler_name in scheduler_names
        )
        print(iteration_results)

    print_final_results(
        scheduler_names, args.n_iterations, feasible_makespans, infeasible_count, computation_times
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Violin plot for makespans using only feasible samples
    plot_violin(
        axs[0, 0],
        feasible_makespans,
        scheduler_names,
        "makespan",
        f"Makespan Comparison on {args.n_iterations} instances (seed {seed}) of {n_tasks}t{n_robots}r{n_skills}s{n_precedence}p",
    )

    # Violin plot for computation times (all samples)
    plot_violin(
        axs[1, 0],
        computation_times,
        scheduler_names,
        "computation_time",
        "Computation Time Comparison",
    )

    # 1v1 comparison: Greedy vs Sadcher (using all samples)
    compare_makespans_1v1(
        axs[0, 1], makespans["greedy"], makespans["sadcher"], "Greedy", "Sadcher-RT"
    )

    # MILP vs Sadcher comparison (if included)
    if args.including_milp:
        compare_makespans_1v1(
            axs[1, 1], makespans["milp"], makespans["sadcher"], "MILP", "Sadcher-RT"
        )
    else:
        fig.delaxes(axs[1, 1])

    plt.tight_layout()
    plt.show()
