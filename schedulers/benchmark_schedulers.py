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
    n_robots = 2
    n_tasks = 6  
    n_skills = 2
    np.random.seed(1)

    if args.including_milp:
        scheduler_names = ["milp", "greedy", "dbgm", "random_bipartite"]
    else: 
        scheduler_names = ["greedy", "dbgm", "random_bipartite"]


    makespans = {scheduler: [] for scheduler in scheduler_names}
    computation_times = {scheduler: [] for scheduler in scheduler_names}
    infeasible_count = {scheduler: 0 for scheduler in scheduler_names}

    # Track the placements: ranking[scheduler] = [#1st, #2nd, #3rd, 4th]    
    ranking = {scheduler: [0, 0, 0, 0] for scheduler in scheduler_names}

    for iteration in tqdm(range(args.n_iterations)):
        # Generate a problem instance
        problem_instance = generate_random_data(n_tasks, n_robots, n_skills, [])
        #problem_instance = generate_simple_data()
        #problem_instance = generate_simple_homogeneous_data(n_tasks=n_qtasks, n_robots=n_robots)
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
                        checkpoint_path="/home/jakob/thesis/method_explorations/LVWS/checkpoints/gatn_het_no_coal_6t_2r_2s/best_checkpoint.pt",
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
        
        # Rank them for this iteration
        # Build (makespan, scheduler) pairs and sort by makespan
        results = sorted((makespans[scheduler][-1], scheduler) for scheduler in scheduler_names)

        print(results)
        previous_score = None
        previous_rank = 0
        count_processed = 0

        for score, scheduler in results:
            # If tied with the previous score, assign the same rank;
            # otherwise, current rank = (# already processed) + 1.
            if previous_score is None or score != previous_score:
                current_rank = count_processed + 1
            else:
                current_rank = previous_rank

            # Award the scheduler the current rank (convert 1-based rank to 0-based index)
            ranking[scheduler][current_rank - 1] += 1

            previous_score = score
            previous_rank = current_rank
            count_processed += 1
                
            # Current standings
        [print(f"{scheduler.capitalize()} : {ranking[scheduler]}") for scheduler in scheduler_names]

    # Summary
    avg_makespans = {s: np.mean(makespans[s]) for s in scheduler_names}
    avg_computation_times = {s: np.mean(computation_times[s]) for s in scheduler_names}
    print("\nSummary of Results after {iterations} runs:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  1st Place: {ranking[scheduler][0]} times")
        print(f"  2nd Place: {ranking[scheduler][1]} times")
        print(f"  3rd Place: {ranking[scheduler][2]} times")
        print(f"  Average Makespan: {avg_makespans[scheduler]:.2f}")
        print(f"  Average Computation Time: {avg_computation_times[scheduler]:.4f} seconds\n")
        print(f"  Infeasible Count: {infeasible_count[scheduler]}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Violin plot for makespans
    axs[0].violinplot(makespans.values(), showmeans=True)
    axs[0].set_xticks(range(1, len(scheduler_names) + 1))
    axs[0].set_xticklabels(scheduler_names)
    axs[0].set_ylabel("Makespan")
    axs[0].set_title("Makespan Comparison of Different Schedulers")

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.03, len(makespans[scheduler]))  
        axs[0].scatter(np.full_like(makespans[scheduler], i) + x_jitter, makespans[scheduler], alpha=0.5, s=10, color='black')

    # Violin plot for computation times
    axs[1].violinplot(computation_times.values(), showmeans=True)
    axs[1].set_xticks(range(1, len(scheduler_names) + 1))
    axs[1].set_xticklabels(scheduler_names)
    axs[1].set_ylabel("Computation Time (s)")
    axs[1].set_title("Computation Time Comparison of Different Schedulers")

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.03, len(computation_times[scheduler]))  
        axs[1].scatter(np.full_like(computation_times[scheduler], i) + x_jitter, computation_times[scheduler], alpha=0.5, s=10, color='black')

    plt.tight_layout()
    plt.show()