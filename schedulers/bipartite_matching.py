import numpy as np
import pulp
import torch


def solve_bipartite_matching(R, sim, n_threads=6):
    """
    R    : torch.tensor [n_robots, n_tasks], reward matrix
    sim  : Simulation object
    Returns: dict of {(i, j): 0/1} solutions indicating the best A_{i,j}.
    """
    n_robots = len(sim.robots)
    n_tasks = len(sim.tasks)
    n_skills = len(sim.robots[0].capabilities)
    available_robots = [r.robot_id for r in sim.robots if r.available]

    problem = pulp.LpProblem("BipartiteMatching", pulp.LpMaximize)

    # Decision variables: A[robot][task] in {0,1}
    A = pulp.LpVariable.dicts(
        "A", (range(n_robots), range(n_tasks)), lowBound=0, upBound=1, cat=pulp.LpBinary
    )

    # Binary "activated" variables for the bipartite matching --> only checks requirements for tasks that will be scheduled this round
    X = pulp.LpVariable.dicts("X", range(n_tasks), lowBound=0, upBound=1, cat=pulp.LpBinary)

    M_robots = n_robots

    # Objective: maximize total reward
    problem += (
        pulp.lpSum(
            R[robot_idx][task_idx] * A[robot_idx][task_idx]
            for robot_idx in range(n_robots)
            for task_idx in range(n_tasks)
        ),
        "TotalReward",
    )

    for robot_idx, robot in enumerate(sim.robots):
        # Constraint: each available robot can take at most one task
        if robot_idx in available_robots:
            problem += pulp.lpSum(A[robot_idx][task] for task in range(n_tasks)) <= 1

        # unavailable robots cannot take any task
        else:
            for task in range(n_tasks):
                problem += A[robot_idx][task] == 0

    # Subteaming constraints
    for task_idx, task in enumerate(sim.tasks):
        # Only constrain if this task is ready
        if task.ready and task.incomplete:
            # Link A to X with big-M constraints
            problem += (
                pulp.lpSum(A[robot_idx][task_idx] for robot_idx in range(n_robots))
                <= M_robots * X[task_idx]
            )
            problem += (
                pulp.lpSum(A[robot_idx][task_idx] for robot_idx in range(n_robots)) >= X[task_idx]
            )

            # 1 ALL REQUIREMENTS MUST BE MET) Capability requirement: if c_t[j][p] = 1, subteam must have it --> effictively all requirements must be met
            for cap in range(n_skills):
                if task.requirements[cap] != 0:
                    problem += (
                        pulp.lpSum(
                            sim.robots[robot_idx].capabilities[cap] * A[robot_idx][task_idx]
                            for robot_idx in range(n_robots)
                        )
                        >= task.requirements[cap] * X[task_idx]
                    )

        else:
            # If task is not ready, force no assignment
            for robot_idx, robot in enumerate(sim.robots):
                problem += A[robot_idx][task_idx] == 0

    problem.solve(
        pulp.PULP_CBC_CMD(
            msg=False,
            threads=n_threads,
            options=[
                "ratioGap 0",
                "allowableGap 0",
            ],
        )
    )
    solution = {
        (robot_idx, task_idx): int(pulp.value(A[robot_idx][task_idx]))
        for robot_idx in range(n_robots)
        for task_idx in range(n_tasks)
    }

    # To see how good the original reward matrix was, we count how often the optimization gave another result
    # compared to the plain argmax over the reward matrix. This is a measure of how good the network understands the problem.
    argmax_over_reward_matrix = torch.argmax(R, axis=1)

    # Only for availabe robots
    argmax_over_reward_matrix = argmax_over_reward_matrix[available_robots]
    shield_triggered_counter = count_differences(argmax_over_reward_matrix, solution)

    # return solution, shield_triggered_counter
    return solution


def count_differences(pre_shield_solution, post_shield_solution):
    assigned_tasks = [
        task_id for (robot_id, task_id), assigned in post_shield_solution.items() if assigned == 1
    ]
    pre_shield_solution = pre_shield_solution.cpu().numpy()
    # Determine the minimum length to compare elements
    min_len = min(len(assigned_tasks), len(post_shield_solution))
    # Count differences in the overlapping part
    diff_count = np.sum(pre_shield_solution[:min_len] != assigned_tasks[:min_len])

    # Count the extra elements in the longer array
    diff_count += abs(len(pre_shield_solution) - len(assigned_tasks))

    return diff_count
