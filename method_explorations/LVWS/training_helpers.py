import json
import os

import sys
sys.path.append("../..")
from helper_functions.schedules import Full_Horizon_Schedule
from simulation_environment.task_robot_classes import Robot, Task
import numpy as np
import torch 


def get_task_status(solution, task_id,  timestep):
    for robot_id, assignments in solution.items():
        for assigned_task_id, start_time, end_time in assignments:
            if assigned_task_id == task_id:
                if timestep < start_time:
                    # Task is ready but not yet started
                    return {"ready": 1, "assigned": 0, "incomplete": 1}
                elif start_time <= timestep <= end_time:
                    # Task is currently being worked on
                    return {"ready": 1, "assigned": 1, "incomplete": 1}
                elif timestep > end_time:
                    # Task is completed
                    return {"ready": 1, "assigned": 0, "incomplete": 0}
    
    # If the task is not found in the solution
    print(f"Task {task_id} not found in the solution")


def create_task_features_from_optimal(problem_instance, solution,  timestep):
    task_features = []
    for task_id, task_requirements in enumerate(problem_instance["R"][1:-1]): # Exclude start and end task
        xy_location = np.array(problem_instance["task_locations"][task_id + 1])
        duration = problem_instance["T_e"][task_id + 1]
        task = Task(task_id, xy_location, duration, task_requirements)
        task_status = get_task_status(solution, task_id + 1, timestep)
        task.ready = task_status["ready"]
        task.assigned = task_status["assigned"]
        task.incomplete = task_status["incomplete"]
        task_features.append(task.feature_vector())
    task_features = np.array(task_features)

    return torch.tensor(task_features, dtype=torch.float32)
    

def is_idle(solution, robot_id, timestep):
    for t_id, task_start, task_end in solution[robot_id]:
        if task_start <= timestep <= task_end:
            return False
    return True


def create_robot_features_from_optimal(problem_instance, solution, timestep):
    robot_features = []
    for robot_id, robot_capabilities in enumerate(problem_instance["Q"]):
        xy_location = find_robot_position_from_optimal(problem_instance, solution, timestep, robot_id)
        speed = 1.0
        robot = Robot(robot_id, xy_location, speed, robot_capabilities)
        robot.available = 1 if is_idle(solution, robot_id, timestep) else 0
        robot_features.append(robot.feature_vector())
    robot_features = np.array(robot_features)

    return torch.tensor(robot_features, dtype=torch.float32)


def find_distances_relative_to_robot_from_optimal(problem, solution,  timestep, robot_id):
    last_finished_task = None
    last_finished_end = float('-inf')

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


def find_robot_position_from_optimal(problem, solution, timestep, robot_id):
    last_finished_task = None
    last_finished_end = float('-inf')
    for t_id, start, end in solution[robot_id]:
        if start <= timestep <= end:
            # Robot is currently executing a task
            return np.array(problem["task_locations"][t_id])
        
        elif end < timestep and end > last_finished_end:
            # Robot is currently not executing a task
            last_finished_end = end
            last_finished_task = t_id

    if last_finished_task is not None:
        return np.array(problem["task_locations"][last_finished_task])
        
    else:
        # Still at start task
        return np.array(problem["task_locations"][0])


def get_expert_reward(schedule, decision_time, gamma = 0.99, immediate_reward = 10):
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

    E = np.zeros((n_robots, len(task_ids))) 
    X = np.zeros((n_robots, len(task_ids)))

    def is_idle(robot_id, time):
        for t_id, task_start, task_end in schedule[robot_id]:
            if task_start <= time <= task_end:
                return False
        return True


    for robot_id in schedule.keys():
        if is_idle(robot_id, decision_time):
            # Robot task pair is feasible at decision time 
            X[robot_id, :] = 1

        for task_id, start_time, end_time in schedule[robot_id]:
            # Task is completed at end_time
            if start_time >= decision_time:
                # Expert reward is discounted time to completion (task_id-1, because task 0 is the beginning of the mission)
                E[robot_id, task_id-1] = gamma**(end_time - decision_time) * immediate_reward

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
    end_times_of_tasks = np.array([task[end_time_index] for tasks in solution.robot_schedules.values() for task in tasks])
    decision_points = np.unique(end_times_of_tasks)

    # Also beginning of mission is decsision point --> append 0, round up to nearest integer
    return np.ceil(np.append([0],decision_points))