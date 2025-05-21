import sys

sys.path.append("..")
import time

import numpy as np

from helper_functions.schedules import Full_Horizon_Schedule, calculate_traveled_distance
from schedulers.initialize_schedulers import create_scheduler
from schedulers.sadcher import SadcherScheduler
from schedulers.sadcherRL import RLSadcherScheduler
from simulation_environment.simulator_2D import Simulation


def run_one_simulation(
    problem_instance, scheduler_name, checkpoint_path, sampling=False, worst_case_makespan=None
):
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
    duration_per_decision = []
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

        duration_per_decision.append(time.time() - start_time)
        sim.assign_tasks_to_robots(instantaneous_schedule)
        sim.step_until_next_decision_point(filter_triggered=filter_triggered)

        if sim.timestep > worst_case_makespan:
            sim.makespan = worst_case_makespan
            feasible = False
            break

    n_tasks = len(problem_instance["T_e"])
    schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)

    return sim.makespan, feasible, duration_per_decision, schedule


def evaluate_scheduler_in_simulation(
    scheduler_name,
    problem_instance,
    checkpoint_map,
    n_runs=1,
    sampling=False,
    worst_case_makespan=None,
):
    all_makespans = []
    times_per_decision = []
    all_feasible = []
    all_distances = []

    for _ in range(n_runs):
        makespan, feasible, times_per_decision, schedule = run_one_simulation(
            problem_instance,
            scheduler_name,
            checkpoint_map[scheduler_name],
            sampling=sampling,
            worst_case_makespan=worst_case_makespan,
        )

        all_makespans.append(makespan)
        all_feasible.append(feasible)
        times_per_decision.extend(times_per_decision)
        all_distances.append(calculate_traveled_distance(schedule, problem_instance["T_t"]))

    best_run = np.argmin(all_makespans)
    best_makespan = all_makespans[best_run]
    best_distance = all_distances[best_run]
    avg_time_per_decision = np.mean(times_per_decision)
    total_time_solution = np.sum(times_per_decision) * n_runs  # total time for all runs
    feasible = any(all_feasible)

    return (
        best_makespan,
        best_distance,
        avg_time_per_decision,
        total_time_solution,
        feasible,
    )


def get_scheduler_names(args):
    names = ["greedy", "sadcher"]

    # optional adds
    if args.include_milp:
        names.append("milp")
    if args.include_stochastic_IL_sadcher:
        names.append("stochastic_IL_sadcher")
    if args.include_RL_sadcher:
        names.extend(["rl_sadcher", "rl_sadcher_sampling"])
    if args.include_heteromrta:
        names.extend(["heteromrta", "heteromrta_sampling"])

    # enforce preferred ordering
    preferred = [
        "milp",
        "sadcher",
        "stochastic_IL_sadcher",
        "heteromrta",
        "heteromrta_sampling",
        "greedy",
        "rl_sadcher",
        "rl_sadcher_sampling",
    ]
    return [s for s in preferred if s in names]
