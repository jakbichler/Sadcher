import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append("..")
from baselines.aswale_23.MILP_solver import milp_scheduling
from data_generation.problem_generator import generate_random_data_with_precedence
from simulation_environment.simulator_2D import Simulation
from visualizations.benchmark_visualizations import (
    compare_makespans_1v1,
    plot_violin,
    print_final_results,
)


def create_simulation(
    problem_instance, scheduler, checkpoint_path=None, move_while_waiting=False, model_name=None
):
    if scheduler == "sadcher":
        return Simulation(
            problem_instance,
            scheduler,
            checkpoint_path=checkpoint_path,
            debug=False,
            move_while_waiting=move_while_waiting,
            model_name=model_name,
        )
    else:
        return Simulation(problem_instance, scheduler, debug=False)


if __name__ == "__main__":
    n_tasks = 6
    n_robots = 2
    n_skills = 2
    n_precedence = 0
    seed = 1
    np.random.seed(seed)
    model_name = "6t2r2s"
    checkpoint_path = (
        "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_0_6t2r2s/best_checkpoint.pt"
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
        problem_instance = generate_random_data_with_precedence(
            n_tasks, n_robots, n_skills, n_precedence
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
                computation_times[scheduler].append(time.time() - start_time)
                makespans[scheduler].append(optimal_schedule.makespan)
                feasible_makespans[scheduler].append(optimal_schedule.makespan)
            else:
                # Non-MILP olutins need simulation to roll out schedule
                sim = create_simulation(
                    problem_instance,
                    scheduler,
                    checkpoint_path,
                    move_while_waiting=args.move_while_waiting,
                    model_name=model_name,
                )
                start_time = time.time()
                feasible = True
                while not sim.sim_done:
                    sim.step()
                    if sim.timestep > worst_case_makespan:
                        sim.makespan = worst_case_makespan  # No feasible solution found
                        infeasible_count[scheduler] += 1
                        print(
                            f"Scheduler {scheduler} did not find a feasible solution at {iteration}"
                        )
                        feasible = False
                        break

                computation_times[scheduler].append(time.time() - start_time)
                makespans[scheduler].append(sim.makespan)
                if feasible:
                    feasible_makespans[scheduler].append(sim.makespan)

        iteration_results = sorted(
            (makespans[scheduler][-1], scheduler) for scheduler in scheduler_names
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
        "Makespan",
        f"Makespan Comparison on {args.n_iterations} instances (seed {seed}) of {n_tasks}t{n_robots}r{n_skills}s{n_precedence}p",
    )

    # Violin plot for computation times (all samples)
    plot_violin(
        axs[1, 0],
        computation_times,
        scheduler_names,
        "Computation Time (s)",
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
