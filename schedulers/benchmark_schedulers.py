import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from dbgm_scheduler import DBGMScheduler
from greedy_instantaneous_scheduler import GreedyInstantaneousScheduler
from random_bipartite_matching_scheduler import RandomBipartiteMatchingScheduler
from simulation_environment.simulator_2D import Simulation
from data_generation.problem_generator import ProblemData, generate_random_data, generate_simple_data

if __name__ == "__main__":
    n_robots = 2
    n_tasks = 6
    n_skills = 2
    #np.random.seed(232)  
    iterations = 50
    scheduler_names = ["greedy", "dbgm", "random_bipartite"]
    makespans = {scheduler: [] for scheduler in scheduler_names}

    # Track the placements: ranking[scheduler] = [#1st, #2nd, #3rd]
    ranking = {scheduler: [0, 0, 0] for scheduler in scheduler_names}

    for iteration in tqdm(range(iterations)):
        # Generate a problem instance
        # problem_instance = generate_random_data(n_tasks, n_robots, n_skills, [])
        problem_instance = generate_simple_data()

        # For each scheduler, run the simulation
        for scheduler in scheduler_names:
            if scheduler == "dbgm":
                sim = Simulation(
                    problem_instance, 
                    [],
                    scheduler, 
                    checkpoint_path="/home/jakob/thesis/method_explorations/LVWS/checkpoints/with_location_simple_6t_2r_2s_10000ex/checkpoint_epoch_30.pt",
                    debug=False
                )
            else:
                sim = Simulation(problem_instance, [], scheduler)

            while not sim.sim_done:
                sim.step()

            makespans[scheduler].append(sim.makespan)
        
        # Rank them for this iteration
        # Build (makespan, scheduler) pairs and sort by makespan
        results = sorted((makespans[scheduler][-1], scheduler) for scheduler in scheduler_names)
        # The tuple with the smallest makespan is in position 0 => 1st place, position 1 => 2nd place, position 2 => 3rd place
        for place, (_, scheduler) in enumerate(results):
            ranking[scheduler][place] += 1  # increment the appropriate place count

    avg_makespans = {s: np.mean(makespans[s]) for s in scheduler_names}

    print("\nSummary of Results:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  1st Place: {ranking[scheduler][0]} times")
        print(f"  2nd Place: {ranking[scheduler][1]} times")
        print(f"  3rd Place: {ranking[scheduler][2]} times")
        print(f"  Average Makespan: {avg_makespans[scheduler]:.2f}")
        print("")

    fig, ax = plt.subplots()
    ax.boxplot(makespans.values())
    ax.set_xticklabels(makespans.keys())
    ax.set_ylabel("Makespan")
    ax.set_title("Makespan comparison of different schedulers")
    plt.show()
