import sys
import os 
sys.path.append("..")

import numpy as np

from helper_functions.schedules import Instantaneous_Schedule

def greedy_instantaneous_assignment(sim):
    """
    Returns an Instantaneous_Schedule indicating which idle robots should go to which tasks next.
    """
    idle_robots = [r for r in sim.robots if r.current_task is None or r.current_task.status == 'DONE']
    pending_tasks = [t for t in sim.tasks if t.status == 'PENDING']
    
    robot_assignments = {}
    
    for robot in idle_robots:
        best_task = None
        best_remaining_cap = float('inf')
        best_travel_time = float('inf')

        for task in pending_tasks:
            assigned_robots = [r for r in sim.robots if r.current_task == task and r != robot]
            combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
            for r_assigned in assigned_robots:
                combined_capabilities = np.logical_or(combined_capabilities, r_assigned.capabilities)
 
            needed_mask = task.requirements & ~combined_capabilities

            new_coverage = np.logical_or(combined_capabilities, robot.capabilities)
            new_needed_mask = task.requirements & ~new_coverage
            new_remaining_cap = np.sum(new_needed_mask)

            if np.array_equal(needed_mask, new_needed_mask):
                # Robot does not add to this task -> redundant 
                print(f"Robot {robot.robot_id} does not add to task {task.task_id}")
                continue

            travel_time = np.linalg.norm(robot.location - task.location) / robot.speed

            # Greedy selection (reduce most capabilities, break tie by travel_time)
            if (new_remaining_cap < best_remaining_cap) or \
               (new_remaining_cap == best_remaining_cap and travel_time < best_travel_time):
                best_task = task
                best_remaining_cap = new_remaining_cap
                best_travel_time = travel_time

        if best_task:
            robot_assignments[robot.robot_id] = best_task.task_id
            robot.current_task = best_task
        else:
            robot_assignments[robot.robot_id] = None
    
    return Instantaneous_Schedule(robot_assignments)
