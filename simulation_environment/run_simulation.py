import argparse
import sys

import numpy as np
import yaml

sys.path.append("..")
from simulator_2D import Simulation

from data_generation.problem_generator import (
    generate_random_data,
    generate_random_data_with_precedence,
)
from helper_functions.schedules import Full_Horizon_Schedule
from schedulers.initialize_schedulers import create_scheduler
from schedulers.sadcher import SadcherScheduler
from simulation_environment.display_simulation import run_video_mode, visualize
from visualizations.solution_visualization import plot_gantt_and_trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize the simulation")
    parser.add_argument("--video", action="store_true", help="Generate a video of the simulation")
    parser.add_argument(
        "--scheduler", type=str, help="Scheduler to use (greedy or random_bipartite)"
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument(
        "--move_while_waiting",
        action="store_true",
        help="Move robots towards second highest reward task while waiting",
    )
    parser.add_argument("--sadcher_model_name", type=str, help="Name of the model to use")
    args = parser.parse_args()

    with open("simulation_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    n_tasks = config["n_tasks"]
    n_robots = config["n_robots"]
    n_skills = config["n_skills"]
    n_precedence = config["n_precedence"]
    np.random.seed(config["random_seed"])
    precedence_constraints = config["precedence_constraints"]

    # problem_instance = generate_random_data(n_tasks, n_robots, n_skills, precedence_constraints)
    problem_instance = generate_random_data_with_precedence(
        n_tasks, n_robots, n_skills, n_precedence
    )

    sim = Simulation(
        problem_instance,
        scheduler_name=args.scheduler,
        debug=True,
    )

    checkpoint_path = (
        "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
    )

    scheduler = create_scheduler(
        args.scheduler,
        checkpoint_path,
        args.sadcher_model_name,
        duration_normalization=sim.duration_normalization,
        location_normalization=sim.location_normalization,
    )

    if args.video:
        # Step simulation, saving frames each time, then generate .mp4
        run_video_mode(sim)
    elif args.visualize:
        # Interactive mode
        visualize(sim, scheduler)
    else:
        # Run simulation until completion
        while not sim.sim_done:
            if isinstance(scheduler, SadcherScheduler):
                predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
                sim.find_highest_non_idle_reward(predicted_reward)
            else:
                instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
            sim.assign_tasks_to_robots(instantaneous_schedule)
            sim.step_until_next_decision_point()

    rolled_out_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)
    print(rolled_out_schedule)
    print(f"Sum of computation times: {sum(sim.scheduler_computation_times)}")
    plot_gantt_and_trajectories(
        f"{sim.scheduler_name}: MS, {sim.makespan}, \n nt: {n_tasks}, nr: {n_robots}, sn: {n_skills}, seed: {config['random_seed']}",
        rolled_out_schedule,
        problem_instance,
    )
