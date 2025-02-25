import argparse
import sys
sys.path.append('..') 
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from baselines.aswale_23.MILP_solver import milp_scheduling
from dbgm_scheduler import DBGMScheduler
from greedy_instantaneous_scheduler import GreedyInstantaneousScheduler
from random_bipartite_matching_scheduler import RandomBipartiteMatchingScheduler
from simulation_environment.simulator_2D import Simulation
from data_generation.problem_generator import (
    ProblemData, generate_random_data, generate_simple_data,
    generate_simple_homogeneous_data, generate_biased_homogeneous_data,
    generate_heterogeneous_no_coalition_data, generate_idle_data, generate_random_data_with_precedence
)


def plot_violin(ax, data, labels, ylabel, title):
    ax.violinplot(data.values(), showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i, scheduler in enumerate(labels, start=1):
        x_jitter = np.random.normal(0, 0.03, len(data[scheduler]))  
        ax.scatter(np.full_like(data[scheduler], i) + x_jitter, data[scheduler],
                   alpha=0.5, s=10, color='black')


def compare_makespans_1v1(ax, makespans1, makespans2, scheduler1, scheduler2):
    makespans1 = np.array(makespans1)
    makespans2 = np.array(makespans2)    
    min_value = min(min(makespans1), min(makespans2))
    max_value = max(max(makespans1), max(makespans2))

    # Compute deviations from parity line
    delta = makespans2 - makespans1
    scheduler_1_wins = delta[delta > 0]
    scheduler_2_wins = delta[delta < 0]

    # Compute 90th percentiles separately
    scheduler_1_wins_90p = np.percentile(scheduler_1_wins, 90) if len(scheduler_1_wins) > 0 else 0
    scheduler_2_wins_90p = np.percentile(np.abs(scheduler_2_wins), 90) if len(scheduler_2_wins) > 0 else 0

    x_vals = np.linspace(min_value, max_value, 100)
    parity_line = x_vals
    upper_bound = x_vals + scheduler_1_wins_90p
    lower_bound = x_vals - scheduler_2_wins_90p

    ax.scatter(makespans1, makespans2, alpha=0.7)
    ax.plot(parity_line, parity_line, color="black", label="Parity", linestyle="--")
    ax.fill_between(x_vals, parity_line, upper_bound, color="red", alpha=0.2,
                    label=f"{scheduler1} wins, \u03B4_90p = {scheduler_1_wins_90p:.1f}")
    ax.fill_between(x_vals, parity_line, lower_bound, color="green", alpha=0.2,
                    label=f"{scheduler2} wins, \u03B4_90p = {scheduler_2_wins_90p:.1f}") 
    ax.legend()
    ax.set_xlabel(f"{scheduler1} Makespan")
    ax.set_ylabel(f"{scheduler2} Makespan")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--including_milp", default=False, action="store_true",
                            help="Include MILP in the comparison")
    arg_parser.add_argument("--n_iterations", type=int, default=50,
                            help="Number of iterations to run")
    arg_parser.add_argument("--move_while_waiting", default=False, action="store_true",
                            help="Allow robots to move while waiting for tasks")
    args = arg_parser.parse_args()
    n_tasks = 6
    n_robots = 2 
    n_skills = 2
    n_precedence = 2
    np.random.seed(1)

    if args.including_milp:
        scheduler_names = ["milp", "greedy", "dbgm", "random_bipartite"]
    else: 
        scheduler_names = ["greedy", "dbgm", "random_bipartite"]

    makespans = {scheduler: [] for scheduler in scheduler_names}
    # Record only feasible makespans for averages/violin plots
    feasible_makespans = {scheduler: [] for scheduler in scheduler_names}
    computation_times = {scheduler: [] for scheduler in scheduler_names}
    infeasible_count = {scheduler: 0 for scheduler in scheduler_names}

    for iteration in tqdm(range(args.n_iterations)):
        # Generate a problem instance
        #problem_instance = generate_random_data(n_tasks, n_robots, n_skills, [])
        problem_instance = generate_random_data_with_precedence(n_tasks, n_robots, n_skills, n_precedence)
        #problem_instance = generate_idle_data()
        #problem_instance = generate_simple_data()
        #problem_instance = generate_simple_homogeneous_data(n_tasks=n_tasks, n_robots=n_robots)
        #problem_instance = generate_biased_homogeneous_data()
        #problem_instance = generate_heterogeneous_no_coalition_data(n_tasks)

        worst_case_makespan = np.sum(problem_instance['T_e']) + \
            np.sum([np.max(problem_instance['T_t'][task]) for task in range(n_tasks + 1)])

        for scheduler in scheduler_names:
            if scheduler == "milp":
                start_time = time.time()
                optimal_schedule = milp_scheduling(problem_instance, n_threads=8,
                                                   cutoff_time_seconds=10 * 60) 
                comp_time = time.time() - start_time
                computation_times[scheduler].append(comp_time)
                ms = optimal_schedule.makespan
                makespans[scheduler].append(ms)
                # Assume MILP always returns a feasible solution
                feasible_makespans[scheduler].append(ms)
            else:
                if scheduler == "dbgm":
                    sim = Simulation(
                        problem_instance, 
                        scheduler, 
                        checkpoint_path="/home/jakob/thesis/method_explorations/DBGM/checkpoints/researching_precedence/RANDOM11_FineTune_80k_6t2r2s2p/best_checkpoint.pt",
                        debug=False,
                        move_while_waiting=args.move_while_waiting
                    )
                else:
                    sim = Simulation(problem_instance, scheduler, debug=False)

                start_time = time.time()
                feasible = True
                while not sim.sim_done:
                    sim.step()
                    if sim.timestep > worst_case_makespan:
                        sim.makespan = worst_case_makespan  # No feasible solution found
                        infeasible_count[scheduler] += 1
                        feasible = False
                        break

                comp_time = time.time() - start_time
                computation_times[scheduler].append(comp_time)
                ms = sim.makespan
                makespans[scheduler].append(ms)
                if feasible:
                    feasible_makespans[scheduler].append(ms)
        
        results = sorted((makespans[scheduler][-1], scheduler) for scheduler in scheduler_names)
        print(results)

    # Averages computed only over feasible samples
    avg_makespans = {s: np.mean(feasible_makespans[s]) if feasible_makespans[s] else float('nan')
                     for s in scheduler_names}
    avg_computation_times = {s: np.mean(computation_times[s]) for s in scheduler_names}
    print(f"\nSummary of Results after {args.n_iterations} runs:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  Average Makespan (feasible only): {avg_makespans[scheduler]:.2f}")
        print(f"  Average Computation Time: {avg_computation_times[scheduler]:.4f} seconds")
        print(f"  Infeasible Count: {infeasible_count[scheduler]}\n")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  

    # Violin plot for makespans using only feasible samples
    plot_violin(axs[0, 0], feasible_makespans, scheduler_names, "Makespan",
                "Makespan Comparison (Feasible Only)")

    # Violin plot for computation times (all samples)
    plot_violin(axs[1, 0], computation_times, scheduler_names, "Computation Time (s)",
                "Computation Time Comparison")

    # 1v1 comparison: Greedy vs DBGMScheduler (using all samples)
    compare_makespans_1v1(axs[0, 1], makespans["greedy"], makespans["dbgm"],
                          "Greedy", "DBGMScheduler")

    # MILP vs DBGM comparison (if included)
    if args.including_milp:
        compare_makespans_1v1(axs[1, 1], makespans["milp"], makespans["dbgm"],
                              "MILP", "DBGMScheduler")
    else:
        fig.delaxes(axs[1, 1])  

    plt.tight_layout()
    plt.show()
