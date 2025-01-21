import numpy as np
from .bigraph_matching import solve_bipartite_matching
from helper_functions.schedules import Instantaneous_Schedule

def random_bipartite_assignment(sim):
    n_robots = len(sim.robots)
    n_tasks = len(sim.tasks)
    robot_assignments = {}
    # Special case for the last task
    idle_robots = [r for r in sim.robots if r.current_task is None or r.current_task.status == 'DONE']
    pending_tasks = [t for t in sim.tasks if t.status == 'PENDING']
    # Check if all tasks are done -> send all robots to the exit task
    if len(pending_tasks) == 1:
        for robot in idle_robots:
            robot_assignments[robot.robot_id] = pending_tasks[0].task_id
            robot.current_task = pending_tasks[0]
        return Instantaneous_Schedule(robot_assignments)

    # Create random reward matrix 
    R = np.random.randint(1, 10, size=(n_robots, n_tasks))
    bipartite_matching_solution = solve_bipartite_matching(R, sim)

    robot_assignments = {robot: task for (robot, task), val in bipartite_matching_solution.items() if val == 1}

    return Instantaneous_Schedule(robot_assignments)