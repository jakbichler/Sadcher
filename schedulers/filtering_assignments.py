from collections import defaultdict

import numpy as np


def filter_unqualified_assignments(assignment_solution, sim):
    """
    Remove any (robot,task) assignment where the robot has
    no overlap with the task's requirements.
    """
    filtered = dict(assignment_solution)
    for (robot_id, task_id), val in assignment_solution.items():
        if val != 1:
            continue
        task = sim.tasks[task_id]
        # skip tasks with no requirements (e.g. idle)
        if not np.any(task.requirements):
            continue
        robot = sim.robots[robot_id]
        if not np.any(robot.capabilities & task.requirements):
            filtered[(robot_id, task_id)] = 0

    return filtered


def filter_redundant_assignments(assignment_solution, sim):
    """
    If a new assignment doesn't add any new skills beyond what's already
    provided by the *existing set* of assigned robots, remove it.
    """
    filtered_solution = dict(assignment_solution)  # copy so we can modify
    for (robot_id, task_id), val in assignment_solution.items():
        if val == 1:
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
            for robot in sim.robots:
                if robot.current_task == task:
                    test_coverage |= robot.capabilities

            for other_rid in used_new_robots:
                if other_rid != robot_id:
                    test_coverage |= sim.robots[other_rid].capabilities

            # If still fully covered, remove this robot
            if np.all(test_coverage[task.requirements]):
                filtered_solution[(robot_id, task_id)] = 0
                used_new_robots.remove(robot_id)

    return filtered_solution


def distance(loc1, loc2):
    return np.linalg.norm(loc1 - loc2)
