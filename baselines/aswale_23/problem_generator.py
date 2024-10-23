from typing import TypedDict

import numpy as np


class ProblemData(TypedDict):
    Q: np.ndarray
    R: np.ndarray
    T_e: np.ndarray
    T_t: np.ndarray


def generate_random_data(n_tasks: int, n_robots: int, n_skills: int) -> ProblemData:
    # Generate random data
    # Q[i][s] = 1 if robot i has skill s, 0 otherwise
    Q = np.random.randint(0, 2, (n_robots, n_skills))

    # Ensure each robot has at least one skill
    for i in range(n_robots):
        if np.sum(Q[i]) == 0:
            Q[i][np.random.randint(0, n_skills)] = 1

    # # Ensure all skills are present in the 
    # for s in range(n_skills):
    #     if np.sum(Q[:, s]) == 0:
    #         Q[np.random.randint(0, n_robots)][s] = 1

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

    # Task travel times
    T_t = np.random.randint(1, 10, (n_tasks + 2, n_tasks + 2))

    return ProblemData(Q=Q, R=R, T_e=T_e, T_t=T_t)