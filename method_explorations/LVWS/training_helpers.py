import json
import os

import sys
sys.path.append("../..")
from helper_functions.schedules import Full_Horizon_Schedule
import numpy as np
import torch 


class Robot:
    def __init__(self, capabilities):
        self.capability = np.array(capabilities)  
        self.available = 1  # 0 if not available, 1 if available

    def feature_vector(self):
        return np.concatenate([self.capability, np.array([self.available])])


class Task:
    def __init__(self, skills_required):
        self.skills_required = np.array(skills_required)  
        self.ready = 1 # 0 if not ready, 1 if ready --> predecessor tasks are finished
        self.assigned = 0 # 0 if not assigned, 1 if assigned
        self.incomplete = 1 # 0 if completed, 1 if incomplete  

    def feature_vector(self):
        return np.concatenate([self.skills_required, np.array([self.ready, self.assigned, self.incomplete])])


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
        task = Task(task_requirements)
        task_status = get_task_status(solution, task_id + 1, timestep)
        task.ready = task_status["ready"]
        task.assigned = task_status["assigned"]
        task.incomplete = task_status["incomplete"]
        task_features.append(task.feature_vector())
    return torch.tensor(task_features, dtype=torch.float32)
    

def is_idle(solution, robot_id, timestep):
    for t_id, task_start, task_end in solution[robot_id]:
        if task_start <= timestep <= task_end:
            return False
    return True


def create_robot_features_from_optimal(problem_instance, solution, timestep):
    robot_features = []
    for robot_id, robot_capabilities in enumerate(problem_instance["Q"]):
        robot = Robot(robot_capabilities)
        robot.available = 1 if is_idle(solution, robot_id, timestep) else 0
        robot_features.append(robot.feature_vector())
    
    return torch.tensor(robot_features, dtype=torch.float32)


def get_expert_reward(schedule, decision_time, gamma = 0.99):
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
                E[robot_id, task_id-1] = gamma**(end_time - decision_time)
                

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

    # Also beginning of mission is decsision point --> append 0
    return np.ceil(np.append([0],decision_points))