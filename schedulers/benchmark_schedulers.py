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
from data_generation.problem_generator import ProblemData, generate_random_data, generate_simple_data, generate_simple_homogeneous_data, generate_biased_homogeneous_data, generate_heterogeneous_no_coalition_data

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--including_milp", default=False,  action="store_true", help="Include MILP in the comparison")
    arg_parser.add_argument("--n_iterations", type=int, default=50, help="Number of iterations to run")
    args = arg_parser.parse_args()
    n_tasks = 6
    n_robots = 2 
    n_skills = 2
    #np.random.seed(1)

    if args.including_milp:
        scheduler_names = ["milp", "greedy", "dbgm", "random_bipartite"]
    else: 
        scheduler_names = ["greedy", "dbgm", "random_bipartite"]

    makespans = {scheduler: [] for scheduler in scheduler_names}
    computation_times = {scheduler: [] for scheduler in scheduler_names}
    infeasible_count = {scheduler: 0 for scheduler in scheduler_names}

    for iteration in tqdm(range(args.n_iterations)):
        # Generate a problem instance
        problem_instance = generate_random_data(n_tasks, n_robots, n_skills, [])
        #problem_instance = generate_simple_data()
        #problem_instance = generate_simple_homogeneous_data(n_tasks=n_tasks, n_robots=n_robots)
        #problem_instance = generate_biased_homogeneous_data()
        #problem_instance = generate_heterogeneous_no_coalition_data(n_tasks)

        worst_case_makespan = np.sum(problem_instance['T_e']) + np.sum([np.max(problem_instance['T_t'][task]) for task in range(n_tasks + 1)])

        # For each scheduler, run the simulation
        for scheduler in scheduler_names:
            if scheduler == "milp":
                start_time = time.time()
                optimal_schedule = milp_scheduling(problem_instance, n_threads=8, cutoff_time_seconds=10 * 60) 
                computation_times[scheduler].append(time.time() - start_time)
                makespans[scheduler].append(optimal_schedule.makespan)            
            
            else:
                if scheduler == "dbgm":
                    sim = Simulation(
                        problem_instance, 
                        [],
                        scheduler, 
                        checkpoint_path="/home/jakob/thesis/method_explorations/LVWS/checkpoints/145k_samples_gatn_with_durations_normalization_per_instance_random_6t_2r_2s/best_checkpoint.pt",
                        debug=False
                    )
                
                else:
                    sim = Simulation(problem_instance, [], scheduler, debug=False)

                start_time = time.time()

                while not sim.sim_done:
                    sim.step()
                    if sim.timestep > worst_case_makespan:
                        sim.makespan = worst_case_makespan * 3 #No feasible solution found
                        infeasible_count[scheduler] += 1
                        break

                computation_times[scheduler].append(time.time() - start_time)
                makespans[scheduler].append(sim.makespan)
        
        results = sorted((makespans[scheduler][-1], scheduler) for scheduler in scheduler_names)
        print(results)

    # Summary
    avg_makespans = {s: np.mean(makespans[s]) for s in scheduler_names}
    avg_computation_times = {s: np.mean(computation_times[s]) for s in scheduler_names}
    print("\nSummary of Results after {iterations} runs:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  Average Makespan: {avg_makespans[scheduler]:.2f}")
        print(f"  Average Computation Time: {avg_computation_times[scheduler]:.4f} seconds\n")
        print(f"  Infeasible Count: {infeasible_count[scheduler]}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  

    # Violin plot for makespans
    axs[0, 0].violinplot(makespans.values(), showmeans=True)
    axs[0, 0].set_xticks(range(1, len(scheduler_names) + 1))
    axs[0, 0].set_xticklabels(scheduler_names)
    axs[0, 0].set_ylabel("Makespan")
    axs[0, 0].set_title("Makespan Comparison")

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.03, len(makespans[scheduler]))  
        axs[0, 0].scatter(np.full_like(makespans[scheduler], i) + x_jitter, makespans[scheduler], alpha=0.5, s=10, color='black')


    # Violin plot for computation times
    axs[1, 0].violinplot(computation_times.values(), showmeans=True)
    axs[1, 0].set_xticks(range(1, len(scheduler_names) + 1))
    axs[1, 0].set_xticklabels(scheduler_names)
    axs[1, 0].set_ylabel("Computation Time (s)")
    axs[1, 0].set_title("Computation Time Comparison")

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.03, len(computation_times[scheduler]))  
        axs[1, 0].scatter(np.full_like(computation_times[scheduler], i) + x_jitter, computation_times[scheduler], alpha=0.5, s=10, color='black')


    # Direct comparison: Greedy vs DBGMScheduler
    min_value = min(min(makespans["greedy"]), min(makespans["dbgm"]))
    max_value = max(max(makespans["greedy"]), max(makespans["dbgm"]))
    axs[0, 1].scatter(makespans["dbgm"], makespans["greedy"], label="DBGMScheduler vs Greedy", alpha=0.7)
    axs[0, 1].plot([min_value, max_value], [min_value, max_value], 'r--', label="x = y", linewidth=2)
    axs[0, 1].set_xlabel("DBGMScheduler Makespan")
    axs[0, 1].set_ylabel("Greedy Makespan")
    axs[0, 1].set_title("Direct Comparison")
    axs[0, 1].legend()


    # MILP  vs DBGM
    if args.including_milp:
        min_value = min(min(makespans["dbgm"]), min(makespans["milp"]))
        max_value = max(max(makespans["dbgm"]), max(makespans["milp"]))
        axs[1, 1].scatter(makespans["dbgm"], makespans["milp"], label="DBGMScheduler vs MILP", alpha=0.7)
        axs[1, 1].plot([min_value, max_value], [min_value, max_value], 'r--', label="x = y", linewidth=2)
        axs[1, 1].set_xlabel("DBGMScheduler Makespan")
        axs[1, 1].set_ylabel("MILP Makespan")
        axs[1, 1].legend()
    else:
        fig.delaxes(axs[1, 1])  

    plt.tight_layout()
    plt.show()