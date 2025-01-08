from typing import TypedDict

import numpy as np


class ProblemData(TypedDict):
    Q: np.ndarray
    R: np.ndarray
    T_e: np.ndarray
    T_t: np.ndarray
    task_locations: np.ndarray
    precedence_constraints: np.ndarray


def generate_random_data(n_tasks: int, n_robots: int, n_skills: int, precedence_constraints) -> ProblemData:
    # Generate random data
    # Q[i][s] = 1 if robot i has skill s, 0 otherwise
    Q = np.random.randint(0, 2, (n_robots, n_skills))

    # Ensure each robot has at least one skill
    for i in range(n_robots):
        if np.sum(Q[i]) == 0:
            Q[i][np.random.randint(0, n_skills)] = 1

    # # Ensure all skills are present in the team
    for s in range(n_skills):
        if np.sum(Q[:, s]) == 0:
            Q[np.random.randint(0, n_robots)][s] = 1

    # R[k][s] = 1 if task k requires skill s, 0 otherwise
    R = np.random.randint(0, 2, (n_tasks, n_skills))

    # Ensure each task requires at least one skill
    for k in range(n_tasks):
        if np.sum(R[k]) == 0:
            R[k][np.random.randint(0, n_skills)] = 1

    # Append start and end tasks
    R = np.vstack([np.zeros(n_skills), R, np.zeros(n_skills)])

    # Task execution times
    T_e = np.random.randint(10, 200, n_tasks) 

    # Append start and end tasks
    T_e = np.hstack([[0], T_e, [0]])

    # Task locations
    grid_size = 100
    task_locations = np.random.randint(0, grid_size,(n_tasks + 2, 2))

    # Travel times between tasks (appr)
    T_t = np.linalg.norm(task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2)

    return ProblemData(Q=Q, R=R, T_e=T_e, T_t=T_t, task_locations=task_locations, precedence_constraints=precedence_constraints)

def generate_simple_data() -> ProblemData:
    """
    Generates a simple problem instance for testing multi-robot task allocation.
    
    Returns:
        ProblemData: A dictionary containing:
            - Q (np.ndarray): Robot skill matrix indicating which robot has which skill.
            - R (np.ndarray): Task requirements matrix indicating required skills for tasks.
            - T_e (np.ndarray): Task execution times including start and end tasks.
            - T_t (np.ndarray): Travel times between tasks based on Euclidean distance.
            - task_locations (np.ndarray): Randomly generated task locations on a grid.
            - precedence_constraints (np.ndarray): Task precedence constraints as an array of (task_j, task_k) pairs.
    """
    n_tasks = 4
    n_skills = 2

    task_type_1 = np.array([1, 0])
    task_type_2 = np.array([0, 1])
    task_type_3 = np.array([1, 1])

    Q = np.array([[1, 0],  # Robot 0 has skill 0
                  [0, 1]])  # Robot 1 has skill 1

    random_tasks = np.array([task_type_1, task_type_2])
    R = np.vstack([
        task_type_3,  # Task type 3 (always present)
        random_tasks[np.random.choice(len(random_tasks), 3, replace=True)]  # Randomly choose 3 tasks
    ])
    R = np.vstack([np.zeros(n_skills), R, np.zeros(n_skills)])

    T_e = np.random.randint(10, 100, n_tasks)
    T_e = np.hstack([[0], T_e, [0]])

    grid_size = 100
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))
    T_t = np.linalg.norm(task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2).round(1)

    precedence_constraints = np.array([])

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints
    )

def read_problem_instance(problem_instance: ProblemData):
    Q = problem_instance['Q']
    R = problem_instance['R']
    T_e = problem_instance['T_e']
    T_t = problem_instance['T_t']
    task_locations = problem_instance['task_locations']
    precedence_constraints = problem_instance['precedence_constraints']
    return Q, R, T_e, T_t, task_locations, precedence_constraints
