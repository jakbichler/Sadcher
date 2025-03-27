from collections import defaultdict

import numpy as np
import pulp
import torch


def solve_bipartite_matching(R, sim):
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
            threads=6,
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


def filter_redundant_assignments(assignment_solution, sim):
    """
    If a new assignment doesn't add any new skills beyond what's already
    provided by the *existing set* of assigned robots, remove it.
    """
    filtered_solution = dict(assignment_solution)  # copy so we can modify

    for (robot_id, task_id), val in assignment_solution.items():
        if val == 1:
            # Find any robots currently assigned to this task
            existing_robots = [r for r in sim.robots if r.current_task == sim.tasks[task_id]]
            # If at least one robot is already on this task,
            # check if their combined capabilities cover all requirements.
            if len(existing_robots) > 0:
                task = sim.tasks[task_id]
                combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
                for rb in existing_robots:
                    combined_capabilities = np.logical_or(combined_capabilities, rb.capabilities)

                # If all requirements are covered by the existing sub-team:
                if np.all(combined_capabilities[task.requirements]):
                    # Then this new assignment doesn't add value; remove it.
                    if np.all(sim.tasks[task_id].requirements == 0):
                        continue  # Idle task should not be filtered
                    else:
                        filtered_solution[(robot_id, task_id)] = 0

    return filtered_solution


def filter_overassignments(assignment_solution, sim):
    """
    If in the current instantaneous assignment, a robot does not add anything that is already covered
    by the other assignments at this moment, only assign the necessary robots
    """
    filtered_solution = dict(assignment_solution)
    task_to_new = defaultdict(list)

    # Group new assignments by task
    for (robot_id, task_id), val in assignment_solution.items():
        if val == 1:
            task_to_new[task_id].append(robot_id)

    for task_id, new_robot_ids in task_to_new.items():
        task = sim.tasks[task_id]

        # If this task is the Idle task or it’s not ready or not incomplete, skip
        if np.all(task.requirements == 0) or not (task.ready and task.incomplete):
            continue

        # Sort the new_robot_ids by distance to the task location -> assign the closest robots first
        new_robot_ids.sort(key=lambda rid: distance(sim.robots[rid].location, task.location))

        # Coverage from existing robots
        combined_caps = np.zeros_like(task.requirements, dtype=bool)
        for robot in sim.robots:
            if robot.current_task == task:
                combined_caps |= robot.capabilities

        used_new_robots = []
        for robot_id in new_robot_ids:
            if np.all(combined_caps[task.requirements]):
                filtered_solution[(robot_id, task_id)] = 0
            else:
                used_new_robots.append(robot_id)
                combined_caps |= sim.robots[robot_id].capabilities

        # Remove any new robot that’s not strictly needed by checking if team coverage remains full if it’s removed
        for robot_id in used_new_robots:
            test_coverage = np.zeros_like(task.requirements, dtype=bool)
            # Add existing coverage
            for robot in sim.robots:
                if robot.current_task == task:
                    test_coverage |= robot.capabilities

            # Add all other new robots
            for other_rid in used_new_robots:
                if other_rid != robot_id:
                    test_coverage |= sim.robots[other_rid].capabilities

            # If still fully covered, remove this robot
            if np.all(test_coverage[task.requirements]):
                filtered_solution[(robot_id, task_id)] = 0
                used_new_robots.remove(robot_id)

    return filtered_solution


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


def distance(loc1, loc2):
    return np.linalg.norm(loc1 - loc2)
