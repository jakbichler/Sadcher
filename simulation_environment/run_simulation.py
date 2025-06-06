import argparse
import json
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
from schedulers.sadcherRL import RLSadcherScheduler
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
        "--start_end_identical",
        action="store_true",
        help="Whether the start and end depot should be in the sm location (to compare against MARMOTLAB)",
    )
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

    if args.start_end_identical:
        problem_instance["task_locations"][-1] = problem_instance["task_locations"][0].copy()

    # problem_instance = json.load(
    # open(
    # "/home/jakob/thesis/datasets/delft_blue_8t3r3s/8t3r3s1p/8t3r3s1p/problem_instances/problem_instance_001422.json"
    # )
    # )

    if args.scheduler == "rl_sadcher":
        sim = Simulation(
            problem_instance, scheduler_name=args.scheduler, debug=True, use_idle=False
        )
    else:
        sim = Simulation(problem_instance, scheduler_name=args.scheduler, debug=True, use_idle=True)

    checkpoint_path = "/home/jakob/thesis/reinforcement_learning/archived_runs/revisit_discrete/25-05-15_all_instances_frozen_encoders/checkpoints/best_agent.pt"

    scheduler = create_scheduler(
        args.scheduler,
        checkpoint_path,
        duration_normalization=sim.duration_normalization,
        location_normalization=sim.location_normalization,
        debugging=args.debug,
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
            filter_triggered = False
            if isinstance(scheduler, SadcherScheduler):
                predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
                sim.find_highest_non_idle_reward(predicted_reward)
            elif isinstance(scheduler, RLSadcherScheduler):
                instantaneous_schedule, filter_triggered = scheduler.calculate_robot_assignment(
                    sim, sampling=False
                )
            else:
                instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
            sim.assign_tasks_to_robots(instantaneous_schedule)
            sim.step_until_next_decision_point(filter_triggered=filter_triggered)

    rolled_out_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)
    print(rolled_out_schedule)
    print(f"Sum of computation times: {sum(sim.scheduler_computation_times)}")
    plot_gantt_and_trajectories(
        f"{sim.scheduler_name}: MS, {sim.makespan}, \n nt: {n_tasks}, nr: {n_robots}, sn: {n_skills}, seed: {config['random_seed']}",
        rolled_out_schedule,
        problem_instance,
    )
