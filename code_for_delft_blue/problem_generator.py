from typing import TypedDict
import numpy as np

class ProblemData(TypedDict):
    Q: np.ndarray
    R: np.ndarray
    T_e: np.ndarray
    T_t: np.ndarray
    task_locations: np.ndarray
    precedence_constraints: np.ndarray


def generate_random_data(n_tasks: int, n_robots: int, n_skills: int,precedence_constraints = None) -> ProblemData:
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
    T_e = np.random.randint(50, 100, n_tasks) 

    # Append start and end tasks
    T_e = np.hstack([[0], T_e, [0]])

    # Task locations
    grid_size = 100
    task_locations = np.random.randint(0, grid_size,(n_tasks + 2, 2))

    # Travel times between tasks (appr)
    T_t = np.linalg.norm(task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2).round(0)

    return ProblemData(Q=Q, R=R, T_e=T_e, T_t=T_t, task_locations=task_locations, precedence_constraints=precedence_constraints)


def generate_random_data_with_precedence(
    n_tasks: int,
    n_robots: int,
    n_skills: int,
    n_precedence: int = 0
) -> ProblemData:
    # ------------------------
    # 1) Same random data generation as generate_random_data
    # ------------------------
    # Q[i][s] = 1 if robot i has skill s, 0 otherwise
    Q = np.random.randint(0, 2, (n_robots, n_skills))
    # Ensure each robot has at least one skill
    for i in range(n_robots):
        if np.sum(Q[i]) == 0:
            Q[i][np.random.randint(0, n_skills)] = 1
    # Ensure every skill is present somewhere
    for s in range(n_skills):
        if np.sum(Q[:, s]) == 0:
            Q[np.random.randint(0, n_robots), s] = 1

    # R[k][s] = 1 if task k requires skill s, 0 otherwise
    R = np.random.randint(0, 2, (n_tasks, n_skills))
    # Ensure each task requires at least one skill
    for k in range(n_tasks):
        if np.sum(R[k]) == 0:
            R[k, np.random.randint(0, n_skills)] = 1

    # Append start(0) and end(n_tasks+1) rows to R
    R = np.vstack([np.zeros(n_skills), R, np.zeros(n_skills)])

    # Task execution times
    T_e = np.random.randint(50, 100, n_tasks)
    # Append 0 for start/end
    T_e = np.hstack([[0], T_e, [0]])

    # Task locations
    grid_size = 100
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))

    # Travel times
    T_t = np.linalg.norm(
        task_locations[:, np.newaxis] - task_locations[np.newaxis, :],
        axis=2
    )

    # ------------------------
    # 2) Generate random precedence constraints (acyclic)
    # ------------------------

    def generate_precedence_constraints(n_tasks, n_precedence):
        # We only consider the 'internal' tasks 1..n_tasks for random precedences
        # (i.e., skip 0 and n_tasks+1 which are "start" and "end")
        internal_tasks = list(range(1, n_tasks + 1))
        np.random.shuffle(internal_tasks)
        possible_pairs = []
        for idx_i in range(len(internal_tasks)):
            for idx_j in range(idx_i + 1, len(internal_tasks)):
                i_task = internal_tasks[idx_i]
                j_task = internal_tasks[idx_j]
                possible_pairs.append((i_task, j_task))

        
        np.random.shuffle(possible_pairs)
        chosen_constraints = possible_pairs[:n_precedence]
        precedence_constraints = chosen_constraints if chosen_constraints else None
        return precedence_constraints

    precedence_constraints = generate_precedence_constraints(n_tasks, n_precedence)


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
