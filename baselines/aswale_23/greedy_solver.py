"""Implementation inspired by the paper "Heterogeneous Coalition Formation and Scheduling with Multi-Skilled Robots", Aswale 2023
https://arxiv.org/abs/2306.11936

extension of the greedy solver to include precedence constraints
"""

import numpy as np
from data_generation.problem_generator import ProblemData, read_problem_instance
from helper_functions.schedule import Schedule


def greedy_scheduling(problem_instance: ProblemData):

    Q, R_original, T_execution, T_travel, task_locations, precedence_constraints = read_problem_instance(problem_instance)
    R = R_original.copy() # R is modified during the scheduling process
    n_robots = Q.shape[0]
    n_tasks = R.shape[0] - 2
    X = np.zeros((n_robots, n_tasks + 2, n_tasks + 2)) # Robot-task assignment matrix
    Y = np.zeros((n_robots, n_tasks + 2)) # Arrival time of robot i at task k
    Y_max = np.zeros(n_tasks + 2)  # Latest arrival time at each task (= start of execution)

    while not all_tasks_satisfied(Q, X, R, n_tasks, n_robots):
        # Find robot-task pairs with max skills to contribute
        max_contribution_pairs = find_max_contribution_pairs(Q, X, R, n_tasks, n_robots, precedence_constraints)

        if not max_contribution_pairs:
            print("No feasible robot-task pairs found. Check for cycles in precedence constraints.")
            break

        # Select the earliest robot-task pair out of the ones with max contribution
        earliest_robot, selected_task, arrival_time = select_earliest_pair(X, max_contribution_pairs, Y_max, T_execution, T_travel, precedence_constraints)
        
        # Assign task selected_task to robot earliest_robot
        assign_task_to_robot(Q, R, X, Y, earliest_robot, selected_task, T_execution, T_travel, arrival_time)
        
        # While not all skills are covered, keep assigning robots to the task
        while np.any(R[selected_task] > 0):
            # Find robots with max contributions for remaining skills
            max_contribution_robots = find_max_contribution_robots_for_one_task(Q, R[selected_task])

            if not max_contribution_robots:
                print(f"No robots can contribute to the remaining skills of task {selected_task}.")
                break

            # Select earliest robot with max remaining skills to contribute
            earliest_robot, arrival_time = select_earliest_robot(X, Y_max, max_contribution_robots, selected_task, T_travel, T_execution, precedence_constraints)
            
            # Assign task selected_task to robot earliest_robot
            assign_task_to_robot(Q, R, X, Y, earliest_robot, selected_task, T_execution, T_travel, arrival_time)
        
        Y_max[selected_task] = np.max(Y[:, selected_task])

    makespan = calculate_task_end_times(Y_max, T_execution, T_travel, n_tasks)

    robots = range(n_robots)
    tasks = range(n_tasks + 2)
    robot_schedules = {robot: [] for robot in robots}

    for robot in range(n_robots):
        for task in tasks[:-1]:
            # Check if robot visits task
            if any(X[robot][previous_task][task] == 1 for previous_task in tasks if previous_task != task):
                start_time = Y_max[task]
                end_time = start_time + T_execution[task]
                if task != 0 and task != n_tasks + 1:
                    robot_schedules[robot].append((task, start_time, end_time))

    return Schedule(makespan, robot_schedules, n_tasks)


def get_current_task(robot_index, X):
    current = 0
    while True:
        next_tasks = np.where(X[robot_index][current] == 1)[0]
        if len(next_tasks) == 0:
            break
        next_task = next_tasks[0]
        current = next_task

    return current


def all_tasks_satisfied(Q, X, R, n_tasks, n_robots):
    """
    Checks if all tasks have been satisfied, i.e., all required skills for each task
    have been covered by the assigned robots.
    """
    for task in range(1, n_tasks + 1):  
        assigned_robots = [robot for robot in range(n_robots) if np.any(X[robot, :, task] == 1)]
        
        if not assigned_robots:
            return False  # No robots assigned to this task
        
        skills_provided = np.logical_or.reduce(Q[assigned_robots], axis=0)
        
        # Check if all required skills are covered
        if not np.all(skills_provided >= R[task]):
            return False  

    return True


def find_max_contribution_pairs(Q, X, R, n_tasks, n_robots, precedence_constraints):
    max_contribution = 0
    max_contribution_pairs = []
    for robot in range(n_robots):
        for task in range(n_tasks + 2):
            if np.any(X[robot,:,task] == 1):
                continue # Robot already assigned to task

            # Skip tasks whose predecessors are not fully scheduled
            if not predecessors_completed(task, R, precedence_constraints):
                continue

            contribution = R[task] @ Q[robot]
            if contribution > max_contribution:
                max_contribution = contribution
                max_contribution_pairs = [(robot, task)]
            elif contribution == max_contribution and contribution > 0:
                max_contribution_pairs.append((robot, task))

    return max_contribution_pairs


def predecessors_completed(task, R, precedence_constraints):
    predecessors = [j for (j, k) in precedence_constraints if k == task]
    for predecessor in predecessors:
        if np.any(R[predecessor] > 0):
            return False
    return True


def select_earliest_pair(X, pairs, Y_max, T_execution, T_travel, precedence_constraints):
    earliest_time = np.inf
    earliest_robot, earliest_task = None, None

    for pair in pairs:
        robot, task = pair
        current_task = get_current_task(robot, X)
        arrival_time = Y_max[current_task] + T_execution[current_task] + T_travel[current_task][task]
        earliest_possible_start_time_due_to_precedence =  finish_time_all_predecessors(task, T_travel, Y_max, T_execution, precedence_constraints)
        arrival_time = max(arrival_time, earliest_possible_start_time_due_to_precedence)

        if arrival_time < earliest_time:
            earliest_time = arrival_time
            earliest_robot, earliest_task = pair

    return earliest_robot, earliest_task, earliest_time


def finish_time_all_predecessors(task, T_travel, Y_max, T_execution, precedence_constraints):
    predecessors = [j for (j, k) in precedence_constraints if k == task]
    max_finish_time = 0
    for predecessor in predecessors:
        finish_time = Y_max[predecessor] + T_execution[predecessor]

        if finish_time > max_finish_time:
            max_finish_time = finish_time

    return max_finish_time


def assign_task_to_robot(Q, R, X, Y,robot_index, task_index, T_execution, T_travel, arrival_time):
    """
    Assigns the specified task to the specified robot and updates the relevant matrices.
    """
    current_task = get_current_task(robot_index, X)

    # Update the arrival time of robot i at task k
    Y[robot_index][task_index] = arrival_time

    # Assign task k after current_task for robot i
    X[robot_index][current_task][task_index] = 1

    # Update the list of unaddressed skills at task k
    R[task_index] = np.array(R[task_index], dtype=bool) & ~np.array(Q[robot_index], dtype=bool)


def find_max_contribution_robots_for_one_task(Q, task_requirements):
    """
    Finds all robots that can contribute the maximum number of remaining skills for a specific task.
    """
    contributions = Q @ task_requirements  # Match skills with requirements
    max_contribution = np.max(contributions)
    
    if max_contribution == 0:
        return []  # No robot can contribute to the remaining skills

    max_contribution_robot = np.where(contributions == max_contribution)[0].tolist()
    return max_contribution_robot


def select_earliest_robot(X, Y_max, robots, task_index, T_travel, T_execution, precedence_constraints):
    """
    Selects the robot that can reach the specified task the earliest.
    """
    earliest_time = np.inf
    earliest_robot = None

    for robot in robots:
        current_task = get_current_task(robot, X)
        arrival_time = Y_max[current_task] + T_execution[current_task] + T_travel[current_task][task_index]
        earliest_possible_start_time_due_to_precedence =  finish_time_all_predecessors(task_index, T_travel, Y_max, T_execution, precedence_constraints)
        arrival_time = max(arrival_time, earliest_possible_start_time_due_to_precedence)

        if arrival_time < earliest_time:
            earliest_time = arrival_time
            earliest_robot = robot

    return earliest_robot, arrival_time


def calculate_task_end_times(Y_max, T_execution, T_travel, n_tasks):
    task_end_times = [Y_max[task] + T_execution[task] for task in range(1, n_tasks + 1)]
    last_task_finished = np.argmax(task_end_times)
    finish_task = n_tasks + 1
    last_depot_arrival_time = task_end_times[last_task_finished] + T_travel[last_task_finished+1][finish_task]
    print(f"Greedy time to completion: {last_depot_arrival_time}")

    return last_depot_arrival_time