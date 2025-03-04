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


def plot_violin(ax, data, labels, ylabel, title):
    ax.violinplot(data.values(), showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i, scheduler in enumerate(labels, start=1):
        x_jitter = np.random.normal(0, 0.03, len(data[scheduler]))
        ax.scatter(
            np.full_like(data[scheduler], i) + x_jitter,
            data[scheduler],
            alpha=0.5,
            s=10,
            color="black",
        )


def compare_makespans_1v1(ax, makespans1, makespans2, scheduler1, scheduler2):
    makespans1 = np.array(makespans1)
    makespans2 = np.array(makespans2)

    min_value = min(min(makespans1), min(makespans2))
    max_value = max(max(makespans1), max(makespans2))

    # Fill regions
    ax.fill_between(
        [min_value, max_value],
        [min_value, max_value],
        max_value,
        color="red",
        alpha=0.15,
        label=f"{scheduler1} Wins",
    )
    ax.fill_between(
        [min_value, max_value],
        min_value,
        [min_value, max_value],
        color="green",
        alpha=0.15,
        label=f"{scheduler2} Wins",
    )

    # Scatter plot
    ax.scatter(makespans1, makespans2, color="black", alpha=0.7, edgecolor="k")

    # Parity line
    x_vals = np.linspace(min_value, max_value, 100)
    ax.plot(x_vals, x_vals, color="black", linestyle="--", label="Parity Line")

    # Labels and legend
    ax.set_xlabel(f"{scheduler1} Makespan")
    ax.set_ylabel(f"{scheduler2} Makespan")
    ax.legend()


def print_final_results(feasible_makespans, infeasible_count, computation_times):
    # Averages computed only over feasible samples
    avg_makespans = {
        s: np.mean(feasible_makespans[s]) if feasible_makespans[s] else float("nan")
        for s in scheduler_names
    }
    avg_computation_times = {s: np.mean(computation_times[s]) for s in scheduler_names}
    print(f"\nSummary of Results after {args.n_iterations} runs:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  Average Makespan (feasible only): {avg_makespans[scheduler]:.2f}")
        print(f"  Average Computation Time: {avg_computation_times[scheduler]:.4f} seconds")
        print(f"  Infeasible Count: {infeasible_count[scheduler]}\n")


def create_simulation(problem_instance, scheduler, checkpoint_path=None, move_while_waiting=False):
    if scheduler == "sadcher":
        return Simulation(
            problem_instance,
            scheduler,
            checkpoint_path=checkpoint_path,
            debug=False,
            move_while_waiting=move_while_waiting,
        )
    else:
        return Simulation(problem_instance, scheduler, debug=False)


if __name__ == "__main__":
    n_tasks = 8
    n_robots = 3
    n_skills = 3
    n_precedence = 3
    np.random.seed(42)
    checkpoint_path = (
        "/home/jakob/thesis/imitation_learning/checkpoints/8t3r3s_models/model_0/best_checkpoint.pt"
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

    print_final_results(feasible_makespans, infeasible_count, computation_times)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Violin plot for makespans using only feasible samples
    plot_violin(
        axs[0, 0],
        feasible_makespans,
        scheduler_names,
        "Makespan",
        "Makespan Comparison (Feasible Only)",
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
