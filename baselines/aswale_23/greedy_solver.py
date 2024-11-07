"""Implementation inspired by the paper "Heterogeneous Coalition Formation and Scheduling with Multi-Skilled Robots", Aswale 2023
https://arxiv.org/abs/2306.11936
"""

import numpy as np
from problem_generator import ProblemData, generate_random_data
from visualizations import plot_gantt_chart, prepare_data_for_gantt_chart

# Define parameters
n_tasks = 6
n_robots = 4
n_skills = 2

np.random.seed(3245)

robots = range(n_robots)
skills = range(n_skills)
tasks = range(n_tasks + 2) # Add start and end task


def greedy_task_allocation(Q, R, T_execution, T_travel):
    while not all_tasks_satisfied(X, R):

        # Find robot-task pairs with max skills to contribute
        max_contribution_pairs = find_max_contribution_pairs(Q, R)

        # Select the earliest robot-task pair out of the ones with max contribution
        earliest_robot, selected_task = select_earliest_pair(max_contribution_pairs, Y, T_execution, T_travel)
        
        # Assign task selected_task to robot earliest_robot
        assign_task_to_robot(Q, R, X, Y, earliest_robot, selected_task, T_execution, T_travel)
        
        # While not all skills are covered, keep assigning robots to the task
        while np.any(R[selected_task] > 0):
            # Find robots with max contributions for remaining skills
            max_contribution_robots = find_max_contribution_robots_for_one_task(Q, R[selected_task])
            
            # Select earliest robot with max remaining skills to contribute
            earliest_robot = select_earliest_robot(max_contribution_robots, selected_task, Y, T_travel, X)
            
            # Assign task selected_task to robot earliest_robot
            assign_task_to_robot(Q, R, X, Y, earliest_robot, selected_task, T_execution, T_travel)
        
        Y_max[selected_task] = np.max(Y[:, selected_task])

    return X

def get_current_task(robot_index, X):
    current = 0
    while True:
        next_tasks = np.where(X[robot_index][current] == 1)[0]
        if len(next_tasks) == 0:
            break
        next_task = next_tasks[0]
        current = next_task

    return current


def all_tasks_satisfied(X, R):
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


def find_max_contribution_pairs(Q, R):
    max_contribution = 0
    max_contribution_pairs = []
    for robot in range(n_robots):
        for task in range(n_tasks + 2):
            if np.any(X[robot,:,task] == 1):
                continue # Robot already assigned to task

            contribution = R[task] @ Q[robot]
            if contribution > max_contribution:
                max_contribution = contribution
                max_contribution_pairs = [(robot, task)]
            elif contribution == max_contribution and contribution > 0:
                max_contribution_pairs.append((robot, task))

    return max_contribution_pairs


def select_earliest_pair(pairs, Y, T_execution, T_travel):
    earliest_time = np.inf
    earliest_pair = None
    for pair in pairs:
        robot, task = pair
        current_task = get_current_task(robot, X)
        arrival_time = Y_max[current_task] + T_execution[current_task] + T_travel[current_task][task]
        
        if arrival_time < earliest_time:
            earliest_time = arrival_time
            earliest_pair = pair

    return earliest_pair


def assign_task_to_robot(Q, R, X, Y,robot_index, task_index, T_execution, T_travel):
    """
    Assigns the specified task to the specified robot and updates the relevant matrices.
    """
    current_task = get_current_task(robot_index, X)
    
    # Assign task k after current_task for robot i
    X[robot_index][current_task][task_index] = 1

    # Update the arrival time of robot i at task k
    arrival_time = Y_max[current_task] + T_execution[current_task] + T_travel[current_task][task_index]
    Y[robot_index][task_index] = arrival_time

    # Update the list of unaddressed skills at task k
    R[task_index] = np.array(R[task_index], dtype=bool) & ~np.array(Q[robot_index], dtype=bool)


def find_max_contribution_robots_for_one_task(Q, task_requirements):
    """
    Finds all robots that can contribute the maximum number of remaining skills for a specific task.
    """
    contributions = Q @ task_requirements  # Dot product to find contributions
    max_contribution = np.max(contributions)
    
    if max_contribution == 0:
        return []  # No robot can contribute to the remaining skills

    max_contribution_robots = np.where(contributions == max_contribution)[0].tolist()
    return max_contribution_robots


def select_earliest_robot(robots, task_index, Y, T_travel, X):
    """
    Selects the robot that can reach the specified task the earliest.
    """
    earliest_time = np.inf
    earliest_robot = None

    for robot in robots:
        current_task = get_current_task(robot, X)
        arrival_time = Y_max[current_task] + T_execution[current_task] + T_travel[current_task][task_index]
        
        if arrival_time < earliest_time:
            earliest_time = arrival_time
            earliest_robot = robot

    return earliest_robot


if __name__ == "__main__":

    problem_instance: ProblemData = generate_random_data(n_tasks, n_robots, n_skills)

    """
    Q[i][s] = 1 if robot i has skill s, 0 otherwise
    R[k][s] = 1 if task k requires skill s, 0 otherwise
    X_ijk: Robot i attends task k after j (+2 is start and end task)
    Y_ik: Arrival time of robot i at task k
    Y_max_k: Latest arrival time at task k (when execution starts)
    """
    Q = problem_instance['Q']
    R = problem_instance['R']
    R_original = R.copy() # R will be modified during the task allocation process
    T_execution = problem_instance['T_e']
    T_travel = problem_instance['T_t']
    X = np.zeros((n_robots, n_tasks + 2, n_tasks + 2))
    Y = np.zeros((n_robots, n_tasks + 2)) 
    Y_max = np.zeros(n_tasks + 2) 

    # Run the greedy solver on the problem
    X = greedy_task_allocation(Q, R, T_execution, T_travel)

    task_end_times = [Y_max[task] + T_execution[task] for task in range(1, n_tasks + 1)]
    last_task_finished = np.argmax(task_end_times)
    finish_task = n_tasks + 1
    last_depot_arrival_time = task_end_times[last_task_finished] + T_travel[last_task_finished+1][finish_task]
    print(f"Full time to completion: {last_depot_arrival_time}")

    # Visualize the results
    tasks_to_plot, task_colors = prepare_data_for_gantt_chart(robots, tasks, X, Y_max, T_execution)
    plot_gantt_chart("Greedy solution", robots, tasks, tasks_to_plot, task_colors, Q, R_original, n_tasks, skills)

