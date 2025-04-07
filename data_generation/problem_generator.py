from typing import TypedDict

import numpy as np


class ProblemData(TypedDict):
    Q: np.ndarray
    R: np.ndarray
    T_e: np.ndarray
    T_t: np.ndarray
    task_locations: np.ndarray
    precedence_constraints: np.ndarray


def generate_random_data(
    n_tasks: int, n_robots: int, n_skills: int, precedence_constraints=None
) -> ProblemData:
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
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))

    # Travel times between tasks (appr)
    T_t = np.linalg.norm(task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2)

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints,
    )


def generate_random_data_with_precedence(
    n_tasks: int, n_robots: int, n_skills: int, n_precedence: int = 0
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
    T_t = np.linalg.norm(task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2)

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
        precedence_constraints=precedence_constraints,
    )


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
    n_tasks = 6
    n_skills = 2

    task_type_1 = np.array([1, 0])
    task_type_2 = np.array([0, 1])
    task_type_3 = np.array([1, 1])

    Q = np.array(
        [
            [1, 0],  # Robot 0 has skill 0
            [0, 1],
        ]
    )  # Robot 1 has skill 1

    np.random.shuffle(Q)

    random_tasks = np.array([task_type_1, task_type_2])
    R = np.vstack(
        [
            task_type_3,  # Task type 3 (always present)
            random_tasks[
                np.random.choice(len(random_tasks), n_tasks - 1, replace=True)
            ],  # Randomly choose 5 tasks
        ]
    )

    np.random.shuffle(R)
    R = np.vstack([np.zeros(n_skills), R, np.zeros(n_skills)])

    T_e = np.random.randint(50, 100, n_tasks)
    T_e = np.hstack([[0], T_e, [0]])

    grid_size = 100
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))
    T_t = np.linalg.norm(
        task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2
    ).round(0)

    precedence_constraints = np.array([])

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints,
    )


def generate_simple_homogeneous_data(n_tasks: int, n_robots: int) -> ProblemData:
    """
    Generates a simple problem instance for testing multi-robot task allocation with homogeneous robots.

    Args:
        n_tasks (int): Number of tasks to generate.
        n_robots (int): Number of robots to generate.

    Returns:
        ProblemData: A dictionary containing:
            - Q (np.ndarray): Robot skill matrix indicating which robot has which skill.
            - R (np.ndarray): Task requirements matrix indicating required skills for tasks.
            - T_e (np.ndarray): Task execution times including start and end tasks.
            - T_t (np.ndarray): Travel times between tasks based on Euclidean distance.
            - task_locations (np.ndarray): Randomly generated task locations on a grid.
            - precedence_constraints (np.ndarray): Task precedence constraints as an array of (task_j, task_k) pairs.
    """
    Q = np.ones((n_robots, 1))

    R = np.vstack([np.zeros(1), np.ones((n_tasks, 1)), np.zeros(1)])

    T_e = np.random.randint(80, 100, n_tasks)
    T_e = np.hstack([[0], T_e, [0]])

    grid_size = 100
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))
    T_t = np.linalg.norm(
        task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2
    ).round(0)

    precedence_constraints = np.array([])

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints,
    )


def read_problem_instance(problem_instance: ProblemData):
    Q = problem_instance["Q"]
    R = problem_instance["R"]
    T_e = problem_instance["T_e"]
    T_t = problem_instance["T_t"]
    task_locations = problem_instance["task_locations"]
    precedence_constraints = problem_instance["precedence_constraints"]
    return Q, R, T_e, T_t, task_locations, precedence_constraints


def generate_static_data():
    Q = np.array(
        [
            [1, 0],  # Robot 0 has skill 0
            [0, 1],
        ]
    )  # Robot 1 has skill 1

    R = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [0, 0]])

    T_e = np.array([0, 50, 50, 50, 0])

    task_locations = np.array([[30, 30], [10, 90], [90, 10], [90, 90], [50, 50]])

    T_t = np.linalg.norm(
        task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2
    ).round(0)

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=np.array([]),
    )


def generate_biased_homogeneous_data() -> ProblemData:
    """
    Generates a problem instance with 6 tasks (plus start and end) arranged along a line.
    The optimal solution is heavily dependent on the relative distances.
    """
    n_tasks = 6  # number of “real” tasks
    n_robots = 1

    Q = np.ones((n_robots, 1))

    # Tasks require the skill (trivial in this case)
    # Add start (index 0) and end (index n_tasks+1) tasks with no requirements.
    R_tasks = np.ones((n_tasks, 1))
    R = np.vstack([np.zeros((1, 1)), R_tasks, np.zeros((1, 1))])

    # Execution times: set a fixed duration for each task.
    T_e_tasks = np.full(n_tasks, 50)
    T_e = np.hstack([[0], T_e_tasks, [0]])

    # Define a bias in locations.
    # Let start and end be at the ends of the line.
    start_location = np.array([0, 50])
    end_location = np.array([100, 50])
    # Place the tasks evenly between start and end.
    xs = np.linspace(10, 90, n_tasks)
    ys = np.full(n_tasks, 50)  # constant y-coordinate
    task_locations = np.stack([xs, ys], axis=1)

    # add jitter to xs and ys
    task_locations += np.random.normal(0, 10, task_locations.shape)
    np.random.shuffle(task_locations)

    # Combine start, tasks, and end.
    task_locations = np.vstack([start_location, task_locations, end_location])

    # Travel times: Euclidean distance between tasks.
    T_t = np.linalg.norm(
        task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2
    ).round(0)

    precedence_constraints = np.array([])  # No precedence constraints for simplicity.

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints,
    )


def generate_heterogeneous_no_coalition_data(n_tasks) -> ProblemData:
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
    n_skills = 2

    task_type_1 = np.array([1, 0])
    task_type_2 = np.array([0, 1])

    Q = np.array(
        [
            [1, 0],  # Robot 0 has skill 0
            [0, 1],
        ]
    )  # Robot 1 has skill 1
    np.random.shuffle(Q)

    random_tasks = np.array([task_type_1, task_type_2])
    R = random_tasks[
        np.random.choice(len(random_tasks), n_tasks, replace=True)
    ]  # Randomly choose 6 tasks
    np.random.shuffle(R)
    R = np.vstack([np.zeros(n_skills), R, np.zeros(n_skills)])

    T_e = np.random.randint(50, 100, n_tasks)
    T_e = np.hstack([[0], T_e, [0]])

    grid_size = 100
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))
    T_t = np.linalg.norm(
        task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2
    ).round(0)

    precedence_constraints = np.array([])

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints,
    )


def generate_idle_data() -> ProblemData:
    n_skills = 2
    # Define task types
    task_type_1 = np.array([1, 0])
    task_type_2 = np.array([0, 1])
    task_type_3 = np.array([1, 1])

    # Two robots with different skills
    Q = np.array([[1, 0], [0, 1]])
    np.random.shuffle(Q)

    # Define 5 tasks (to be permuted) and add dummy start/end tasks later
    tasks_R = np.array([task_type_1, task_type_2, task_type_3, task_type_1, task_type_2])
    T_e_tasks = np.array([30, 40, 50, 200, 200])
    T_e_tasks = np.array([t + np.random.randint(-20, 50) for t in T_e_tasks])
    T_e_tasks[1] = T_e_tasks[2] + 20 + np.random.randint(-10, 10)
    # Base locations for 7 points: index 0 = start, indices 1-5 = tasks, index 6 = finish
    base_locations = np.array(
        [[50, 10], [55, 15], [45, 15], [50, 40], [55, 90], [45, 90], [50, 95]]
    )

    # Apply random perturbation and rotation around (50,50)
    randomized_locations = base_locations + np.random.uniform(-5, 5, base_locations.shape)
    theta = np.random.uniform(0, 2 * np.pi)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_locations = (randomized_locations - np.array([50, 50])) @ rot_matrix.T + np.array(
        [50, 50]
    )

    # Permute the tasks (indices 0...4) consistently in R, T_e, and task_locations
    n_tasks = tasks_R.shape[0]
    perm = np.random.permutation(n_tasks)
    tasks_R_perm = tasks_R[perm]
    T_e_tasks_perm = T_e_tasks[perm]
    task_locations_tasks = rotated_locations[1:-1]  # tasks locations (exclude start & finish)
    task_locations_tasks_perm = task_locations_tasks[perm]

    # Rebuild full arrays with fixed start and finish
    R_full = np.vstack([np.zeros(n_skills), tasks_R_perm, np.zeros(n_skills)])
    T_e_full = np.hstack([[0], T_e_tasks_perm, [0]])
    task_locations_full = np.vstack(
        [rotated_locations[0], task_locations_tasks_perm, rotated_locations[-1]]
    )
    T_t = np.linalg.norm(task_locations_full[:, None] - task_locations_full[None, :], axis=2)

    return ProblemData(
        Q=Q,
        R=R_full,
        T_e=T_e_full,
        T_t=T_t,
        task_locations=task_locations_full,
        precedence_constraints=None,
    )


def generate_random_data_all_robots_all_skills(
    n_tasks: int, n_robots: int, n_skills: int, precedence_constraints=None
) -> ProblemData:
    # Generate random data
    # Q[i][s] = 1 if robot i has skill s, 0 otherwise
    Q = np.ones((n_robots, n_skills))  # All robots have all skills

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
    T_e = np.random.randint(80, 100, n_tasks)

    # Append start and end tasks
    T_e = np.hstack([[0], T_e, [0]])

    # Task locations
    grid_size = 100
    task_locations = np.random.randint(0, grid_size, (n_tasks + 2, 2))

    # Travel times between tasks (appr)
    T_t = np.linalg.norm(task_locations[:, np.newaxis] - task_locations[np.newaxis, :], axis=2)

    return ProblemData(
        Q=Q,
        R=R,
        T_e=T_e,
        T_t=T_t,
        task_locations=task_locations,
        precedence_constraints=precedence_constraints,
    )
