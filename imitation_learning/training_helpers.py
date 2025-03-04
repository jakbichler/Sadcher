import json
import os
import sys

sys.path.append("../..")
import numpy as np
import torch

from helper_functions.schedules import Full_Horizon_Schedule
from simulation_environment.task_robot_classes import Robot, Task


def predecessors_completed(problem, solution, task_id, timestep):
    precedence_constraints = problem["precedence_constraints"]
    if precedence_constraints is None:
        return True

    predecessors = [j for (j, k) in precedence_constraints if k == task_id]
    for preceding_task in predecessors:
        if not task_is_completed(solution, preceding_task, timestep):
            return False
    return True


def is_idle(solution, robot_id, timestep):
    for t_id, task_start, task_end in solution[robot_id]:
        if task_start <= timestep <= task_end:
            return False
    return True


def task_is_completed(solution, task_id, timestep):
    for robot_id, assignments in solution.items():
        for assigned_task_id, start_time, end_time in assignments:
            if assigned_task_id == task_id:
                if timestep > end_time:
                    return True
    return False


def get_task_status(problem, solution, task_id, timestep):
    for robot_id, assignments in solution.items():
        for assigned_task_id, start_time, end_time in assignments:
            if assigned_task_id == task_id:
                if timestep < start_time:
                    # Task is not yet started
                    ready = 1 if predecessors_completed(problem, solution, task_id, timestep) else 0
                    return {"ready": ready, "assigned": 0, "incomplete": 1}

                elif start_time <= timestep <= end_time:
                    # Task is currently being worked on
                    return {"ready": 1, "assigned": 1, "incomplete": 1}

                elif timestep > end_time:
                    # Task is completed
                    return {"ready": 1, "assigned": 0, "incomplete": 0}

    # If the task is not found in the solution
    print(f"Task {task_id} not found in the solution")


def create_task_features_from_optimal(
    problem_instance, solution, timestep, location_normalization=100, duration_normalization=100
):
    task_features = []
    for task_id, task_requirements in enumerate(
        problem_instance["R"][1:-1]
    ):  # Exclude start and end task
        xy_location = np.array(problem_instance["task_locations"][task_id + 1])
        task_status = get_task_status(problem_instance, solution, task_id + 1, timestep)
        task = Task(task_id, xy_location, problem_instance["T_e"][task_id + 1], task_requirements)
        task.ready = task_status["ready"]
        task.assigned = task_status["assigned"]
        task.incomplete = task_status["incomplete"]
        task_features.append(task.feature_vector(location_normalization, duration_normalization))
    task_features = np.array(task_features)

    return torch.tensor(task_features, dtype=torch.float32)


def create_robot_features_from_optimal(
    problem_instance, solution, timestep, location_normalization=100, duration_normalization=100
):
    robot_features = []
    for robot_id, robot_capabilities in enumerate(problem_instance["Q"]):
        xy_location, remaining_workload = find_robot_position_and_workload_from_optimal(
            problem_instance, solution, timestep, robot_id
        )
        speed = 1.0
        robot = Robot(robot_id, xy_location, speed, robot_capabilities)
        robot.available = 1 if is_idle(solution, robot_id, timestep) else 0
        robot.remaining_workload = remaining_workload
        robot_features.append(robot.feature_vector(location_normalization, duration_normalization))
    robot_features = np.array(robot_features)

    return torch.tensor(robot_features, dtype=torch.float32)


def find_distances_relative_to_robot_from_optimal(problem, solution, timestep, robot_id):
    last_finished_task = None
    last_finished_end = float("-inf")

    for t_id, start, end in solution[robot_id]:
        if start <= timestep <= end:
            # Robot is currently executing a task
            return np.array(problem["T_t"][t_id][1:-1])

        elif end < timestep and end > last_finished_end:
            # Robot is currently not executing a task
            last_finished_end = end
            last_finished_task = t_id

    if last_finished_task is not None:
        # Return relative distances to all other real tasks (excluding start and end tasks)
        return np.array(problem["T_t"][last_finished_task][1:-1])

    else:
        # Still at start task --> return distance relative to start (Task 0)
        return np.array(problem["T_t"][0][1:-1])


def find_robot_position_and_workload_from_optimal(problem, solution, timestep, robot_id):
    last_finished_task = None
    last_finished_end = float("-inf")
    for t_id, start, end in solution[robot_id]:
        if start <= timestep <= end:
            # Robot is currently executing a task
            location = np.array(problem["task_locations"][t_id])
            remaining_workload = end - timestep
            return location, remaining_workload

        elif end < timestep and end > last_finished_end:
            # Robot is currently not executing a task
            last_finished_end = end
            last_finished_task = t_id

    if last_finished_task is not None:
        location = np.array(problem["task_locations"][last_finished_task])
        remaining_workload = 0
        return location, remaining_workload

    else:
        # Still at start task
        location = np.array(problem["task_locations"][0])
        remaining_workload = 0
        return location, remaining_workload


def get_expert_reward(schedule, decision_time, travel_times, gamma=0.99, immediate_reward=10):
    """
    schedule: dict {robot_id: [(task_id, start_time, end_time), ...]}
    decision_time: float
    gamma: float discount factor
    Returns:
      E: Expert reward matrix[n_robots, n_tasks]
      X: Feasibility mask  [n_robots, n_tasks]
    Assumptions:
      - For now no precedence constraints --> tasks are ready or completed
      - Task completion can be inferred from the intervals
      - Robots are identified by keys in `schedule`
      - Tasks are the unique set of all task_ids in all intervals
    """
    n_robots = len(schedule)
    task_ids = sorted({t_id for r_id in schedule for (t_id, _, _) in schedule[r_id]})

    E = np.zeros((n_robots, len(task_ids) + 1))  # +1 for the idle task
    X = np.zeros((n_robots, len(task_ids) + 1))  # +1 for the idle task
    X[:, -1] = 1  # Idle task is always feasible
    travel_times = np.array(travel_times)

    TIME_EPSILON = 0.01
    IMMEDIATE_IDLE_REWARD = 5

    def is_idle(robot_id, time):
        for t_id, task_start, task_end in schedule[robot_id]:
            if task_start <= time <= task_end:
                return False
        return True

    for robot_id in schedule.keys():
        # Normal tasks
        if is_idle(robot_id, decision_time):
            # Robot task pair is feasible at decision time
            X[robot_id, :] = 1

        for task_id, start_time, end_time in schedule[robot_id]:
            # Task is completed at end_time
            if start_time >= decision_time:
                # Expert reward is discounted time to completion (task_id-1, because task 0 is the beginning of the mission)
                E[robot_id, task_id - 1] = gamma ** (end_time - decision_time) * immediate_reward

        # Idle reward for gaps between consecutive tasks
        for i in range(len(schedule[robot_id]) - 1):
            current_task, _, current_end = schedule[robot_id][i]
            next_task, next_start, _ = schedule[robot_id][i + 1]
            t_ij = travel_times[current_task, next_task]
            idle_end = (
                next_start - t_ij
            )  # robot must depart by this time to arrive exactly at next_start
            if idle_end > current_end + TIME_EPSILON and decision_time < idle_end - TIME_EPSILON:
                E[robot_id, -1] = gamma ** (idle_end - decision_time) * IMMEDIATE_IDLE_REWARD

        # First Task Idle
        if len(schedule[robot_id]) == 0:
            continue
        first_task, first_start, _ = schedule[robot_id][0]
        t_01 = travel_times[0, first_task]
        end_of_idle_task = first_start - t_01
        if (
            t_01 < first_start - TIME_EPSILON
            and decision_time < (first_start - t_01) - TIME_EPSILON
        ):
            E[robot_id, -1] = gamma ** (end_of_idle_task - decision_time) * IMMEDIATE_IDLE_REWARD

    return torch.tensor(E), torch.tensor(X)


def load_dataset(problem_dir, solution_dir):
    problems = []
    solutions = []
    # Load all problem instances
    for file_name in sorted(os.listdir(problem_dir)):
        with open(os.path.join(problem_dir, file_name), "r") as f:
            problems.append(json.load(f))

    # Load all solution files
    for file_name in sorted(os.listdir(solution_dir)):
        with open(os.path.join(solution_dir, file_name), "r") as f:
            solutions.append(json.load(f))

    solutions = [Full_Horizon_Schedule.from_dict(solution) for solution in solutions]

    return problems, solutions


def find_decision_points(solution):
    end_time_index = 2
    end_times_of_tasks = np.array(
        [task[end_time_index] for tasks in solution.robot_schedules.values() for task in tasks]
    )
    decision_points = (
        np.unique(end_times_of_tasks) + 0.1
    )  # Small offset, since robot will only be available after task is completed

    # Also beginning of mission is decsision point --> append 0, round up to nearest integer
    return np.ceil(np.append([0], decision_points))
